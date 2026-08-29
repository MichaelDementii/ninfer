**Level of the claim: operator.** The end-to-end figures at the end are an observation, not the
evidence.

## The observation

`resolve_route` in `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` takes the fused TMA
SwiGLU at exactly `T == 1024`:

```cpp
if (tokens == kPrimaryT) { return Nvfp4LinearSwiGluRoute::TmaFusedW4A4; }
return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
```

The kernel accepts any `T % 256 == 0` - `nvfp4_linear_swiglu_w4a4_tma.cu:55-58` rejects only
`tokens < 256 || tokens % 256 != 0`, and nothing else constrains `T`. So every prefill chunk wider
than 1024 falls into
`LinearW4A4Post`, which materialises the `34816 x T` bf16 projection into the arena and reads it
back through a separate `silu_mul` - for a kernel that was ready to take it.

`program_impl.h` sizes each chunk as `std::min(prefill_chunk, end - cursor)`, so with a configured
chunk of 8192 every full chunk is 8192 and only the prompt's remainder is not. On a 33,031-token
prompt that is four chunks on the materialising path that did not need to be there, and a tail of
263 that does.

## Measured effect

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, gcc 13.3, Ubuntu 24.04, Release,
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Base `3a61ef3f`, both arms built in the same build directory.
Artifact `qwen3.8-27b/nvfp4`. Ratios are master / branch, so above 1.000 is faster.

```bash
./build/bench/ninfer_nvfp4_linear_swiglu_bench --policy a4 \
  --t-sweep 256,512,768,1024,1280,2048,4096,8192 --repeat 50
```

| T | master µs | branch µs | speedup | route on master |
|---:|---:|---:|---:|---|
| 256 | 155.6 | 155.6 | **1.000** | materialising |
| 512 | 313.3 | 313.3 | **1.000** | materialising |
| 768 | 411.6 | 411.6 | **1.000** | materialising |
| 1024 | 417.8 | 417.8 | **1.000** | fused, unchanged |
| 1280 | 530.4 | 501.5 | **1.058** | materialising → fused |
| 2048 | 837.7 | 767.7 | **1.091** | materialising → fused |
| 4096 | 1681.4 | 1497.1 | **1.123** | materialising → fused |
| 8192 | 3456.0 | 3086.1 | **1.120** | materialising → fused |

The `1.000` cells are not a noise floor. Below 1280 both arms resolve to the same route and run the
same kernel, so those rows are a built-in control that the change touches nothing the route table
already covered, and the gain begins exactly at the first multiple of 256 above 1024 - which is what
the new predicate registers. Useful throughput over the same points goes 874 → 874 TFLOP/s at 1024
and 845 → 946 at 8192.

The predicate has two clauses, `tokens >= 1024` and `tokens % 256 == 0`, so the second one is worth
sweeping on its own. A separate run over widths straddling the multiples:

| T | 1024 | 1025 | 1152 | 1279 | **1280** | 1281 | 1408 | **1536** | **1792** | **2048** | **2304** | **2560** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T % 256` | 0 | ≠0 | ≠0 | ≠0 | **0** | ≠0 | ≠0 | **0** | **0** | **0** | **0** | **0** |
| speedup | 0.995 | 1.000 | 1.000 | 1.000 | **1.053** | 0.997 | 1.000 | **1.065** | **1.063** | **1.079** | **1.084** | **1.087** |

The gain appears at every accepted width and at no other, including across the adjacent triple
1279 / 1280 / 1281. The two cells that are not exactly unity, 1024 at 0.995 and 1281 at 0.997, are the run-to-run floor
on a route neither arm changes - 1024 is a multiple of 256 but was already fused on both arms, and
1281 is not a multiple at all; 2048 reads 1.079 in this sweep and
1.091 in the one above, which is the same floor seen from the other side.

Worth noting for its own sake: on `master` the materialising path at 1025 costs 598.0 µs against
417.8 for the fused path at 1024, so the current table has a sawtooth at every 256-wide step above
1024. This change removes it at the accepted widths and leaves it in between.


**Re-taken on the current base.** Master moved to `1fc1cb76` while this was in review; both arms
were rebuilt there and both sweeps re-run in one more pass. The operator reads 1.000 / 1.006 / 1.000
/ 1.000 at 256 / 512 / 768 / 1024, then 1.053 at 1280, 1.091 at 2048, 1.130 at 4096 and 1.122 at
8192. The predicate sweep reads 1.000 at every width that is not a multiple of 256, and 1.051 /
1.065 / 1.067 / 1.080 / 1.084 / 1.090 at 1280 / 1536 / 1792 / 2048 / 2304 / 2560. Every cell is
inside the +-1.2% envelope; the one that moved most is 512, which reads 1.006 in this pass and 1.000
in the table above on a width neither arm re-routes, so that is the floor measuring itself.

## Tradeoffs

**Workspace capacity.** The commit edits `nvfp4_linear_swiglu_workspace_capacity_bytes`, and
workspace boundaries are contract, so this was settled by enumeration rather than argument: the
function was called on both arms over a grid of 4560 `(min_tokens, max_tokens)` intervals.

| | |
|---|---:|
| intervals probed | 4560 |
| where the branch asks for **more** | **0** |
| where the branch asks for less | 963 |
| where argument validation changed | 0 |

952 of the reductions are 72,512 bytes - one token of the materialised bf16 projection
(34816 x 2 = 69,632) plus one token of the NVFP4 activation codes and scales
(5120/2 + 5120/16 = 2,880) - because `last_baseline` steps down when `max_tokens` itself now
resolves to the fused route. The other 11 are intervals collapsed to a single accepted width, where
the materialising path becomes unreachable and the whole projection leaves: at 8192 that is 594,018,304 → 23,592,960 bytes, and
34816 x 8192 x 2 = 570,425,344 exactly.

The mechanism behind the zero is worth stating, so the sweep reads as a proof rather than as brute
force: `baseline_workspace_bytes` exceeds `fused_workspace_bytes` at the same width by roughly 25x
(594,018,304 against 23,592,960 at 8192). So whenever the interval still contains a width that
routes to `LinearW4A4Post`, the baseline term is the maximum and the widened fused term cannot
surface. The only intervals where the total moves materially are the eleven with no such width left.

**The engine does not request such an interval today.** The measured GPU workspace peak is identical
on both arms - 152.57 MiB at `--prefill-chunk 1024`, 1.19 GiB at 8192 - which settles it without
reading further code: had the interval been collapsed, 544 MiB would have left the peak. So the win
here is **traffic, not capacity**, and I would rather say that than quote the 544 MiB.

**Numerical quality.** The two routes are not bit-identical and cannot be: the fused epilogue holds
the f32 accumulator and rounds once, the materialising one rounds gate and up to bf16 and multiplies
the rounded values. This does not introduce that difference, it moves which widths see it. Stock
already answers differently at 1024 and at 8192; after this both answer the way 1024 does.

`tests/ops/linear_swiglu/test_nvfp4.cpp` qualifies **both** routes against the FP64 oracle today -
`kA4Cases{5, 48, 49, 128, 1024}` puts 49 and 128 on the materialising path and 1024 on the fused one
- and this change moves neither, so its verdicts are unchanged. It does not cover 1280 and up, which
is exactly what does change route. Running the same test with that list extended, on both arms, with
`NINFER_OP_REPORT_STATS=1`:

| T | route change | master rel_L2 | branch rel_L2 | master gross | branch gross |
|---:|---|---:|---:|---:|---:|
| 5, 48, 49, 128, 1024 | none | 0.111 … 0.0527 | **identical** | 0.639 … 0.373 | **identical** |
| 1280 | → fused | 0.05159 | **0.05126** | 0.336 | **0.307** |
| 2048 | → fused | 0.05011 | **0.04969** | 0.286 | **0.261** |
| 4096 | → fused | 0.052342 | **0.052338** | 0.286 | **0.326** |

Both arms pass the A4 criterion at every width. Two honest readings: every width whose route is
unchanged reproduces to the last printed digit, which is the confinement claim at the numerical
level; and at the re-routed widths the fused path is equal or very slightly closer on relative-L2
but **worse on the gross tail at 4096**, 0.326 against 0.286. So it is not "strictly more accurate".
For scale, the shipped widths reach 0.639 of their own gross limit against 0.326 here, so the new
route sits well inside the envelope already accepted - noting the limit is width-dependent (0.567 at
T=5, 1.266 at 4096), so these are fractions of different absolute bounds.

The comment above that criterion (`tests/ops/linear_swiglu/linear_swiglu_test_common.cpp:28`) is the
frame I am working from, and it is yours:

> The criterion belongs to the activation-compute profile, not the weight storage format or a
> private materialized/fused implementation.

**The lower bound is conservative, and I measured by how much.** `kPrimaryT` is where the fused
route is registered today and I kept it, but I can no longer say it is where the fused path
starts winning, because it is not. A probe that lowers only the route predicate to `kTmaBlockM`,
three runs (`median_us` over 50 after 5 warmup, materialising against fused):

| T | 256 | 512 | 768 | 1024 | 1280 |
|---|---:|---:|---:|---:|---:|
| fused speedup | x1.12 | x1.35 | x1.28 | x1.00 | x1.05 |

The three runs agree to within 0.011 at 256 and to the third decimal at 512 and 768. 1024 reads
0.995-0.999 because both arms already take the fused route there; that row is the control.

The numerics hold at those widths too. `kA4Cases` extended to 256/512/768, both arms:

| T | master rel_L2 | lowered rel_L2 | master gross | lowered gross |
|---:|---:|---:|---:|---:|
| 256 | 0.059000 | **0.058382** | 0.531 | **0.486** |
| 512 | 0.057068 | **0.056556** | 0.531 | **0.486** |
| 768 | 0.054019 | **0.053731** | 0.408 | **0.373** |

The fused route is the closer of the two at all three widths, on both statistics.

What stops the bound from simply moving is not numerics but a second copy of the bound. With only
the route predicate lowered the test fails at 256, 512 and 768 with `exact workspace query/execution
high-water mismatch`: `nvfp4_linear_swiglu_workspace_capacity_bytes` gates its fused term on
`max_tokens >= kPrimaryT` as well, so below 1024 the query and the executed route disagree. Moving
the bound properly means moving both, over a set of widths no test covers on either arm today. That
is a second mechanism and it is not in this change.

**Other execution paths.** The diff is one predicate and one capacity branch in a host-side route
table, and the compiled result says the same thing: `cuobjdump -sass` over the two binaries finds
**0 of 2899 comparable device functions changed** - 2937 enumerated, 38 unmatched by name because
they are concat symbols with internal linkage - with registers, shared memory and spills identical
everywhere. No kernel is compiled differently; only which one a given width reaches.

`ctest -j1` on the rebased base: **94 tests, 0 failed, 1 skipped**, on master and on this branch.
The skip needs a 27B artifact this box does not have. On the old base this suite had one failure,
the one we reported as #105; `fd48e2fa` between the two bases rewrites that test's oracle and it
now passes.

## End-to-end observation

Arms alternating inside each round, one process per point, greedy, four rounds. Percent faster than
master.

| prompt tokens | chunk | prefill | decode | master round spread |
|---:|---:|---:|---:|---:|
| 33 031 | 8192 | **+2.38%** | 0.00% | 0.91% |
| 65 882 | 8192 | **+1.96%** | 0.00% | 0.70% |
| 33 031 | **1024** | **+0.01%** | 0.00% | 0.32% |

**On the chunk.** You wrote when you closed #96 that the 1024 default is deliberate and that the
workspace a wider chunk costs comes out of KV. That is unchanged here and I am not arguing against
it: on this host and artifact the workspace peak is 152.57 MiB at 1024 and 1.19 GiB at 8192, a
1.04 GiB difference that is entirely the pre-existing cost of the wider chunk and is identical on
both arms. What this change does is stop the wider chunk from also paying a route penalty it never
needed to pay. It does not make the wider chunk cheaper and it is not a reason to move the default;
the 1024 row is the control that says so.

At both 8192 points the slowest of the four branch runs is faster than the fastest of the four
master runs, so the separation does not depend on round ordering. At 1024 the two ranges overlap
completely and the four per-round deltas straddle zero (-0.038%, +0.080%, -0.057%, +0.067%): the
default path is not merely reported as unchanged, it is unresolvable.

A `--max-new 512` generation on the 65,882-token prompt, stopping on the stop token after 164
generated tokens, is byte-identical between the arms at **both** chunk widths - 203 bytes, an
eight-field JSON answer block, one run per arm. That is agreement, not proof: greedy decoding absorbs
small numeric differences at the argmax, and the routes are not required to agree.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
- Every µs figure is the benchmark's `median_us` over 50 samples after 5 warmup. Each T-sweep is one
  pass per arm; the overlap between the two sweeps at 1024 and 2048 is the only repetition and puts
  the run-to-run envelope at roughly ±1.2%, so ratios below x1.02 should be read as unity.
- All workspace measurements use `--max-context 131072`, which pins KV capacity explicitly (headroom
  reads 0 B on both arms). They establish that the two arms request identical workspace, not how an
  auto-sized KV policy would respond to a wider chunk.
- I kept `kPrimaryT` as the lower bound. The fused kernel does win below it and is the closer
  of the two against the oracle there - measured, above - so the clause is conservative rather
  than optimal. Moving it needs the capacity query's own copy of the bound to move with it,
  which is a second mechanism and is not in this change.
- The oracle sweep stops at 4096; 8192 was not run through the FP64 oracle, only through the
  benchmark and the product.
- The output gate is one prompt and one run per arm at each chunk width.
- Only `qwen3.8-27b/nvfp4` is measured. `qwen3.6-27b/nvfp4` takes the same route and has strictly
  greater exposure - it binds an NVFP4 `mlp/gate_up {34816,5120}` on every text layer, where the
  measured artifact is NVFP4 only below layer 56 - and I did not measure it.
- I have not looked at whether the same registration gap exists on the FP8 SwiGLU route.

## Questions

1. Is this within scope, and is an external implementation appropriate? It is one commit in one
   file, +10/-4, and I can open a pull request linked to this Issue, or leave the report.
2. Should extending `kA4Cases` in `tests/ops/linear_swiglu/test_nvfp4.cpp` to cover the widths this
   re-routes be part of the change? The repo asks for relevant shapes and execution routes to be
   exercised, and those widths are currently unqualified on either route.
3. Should the lower bound be `kTmaBlockM` rather than `kPrimaryT`? By the measurement above the
   fused route wins from 256 up and is the numerically closer of the two there, so this change as
   written leaves x1.12 to x1.35 on the table at three widths. I left it out because it needs the
   capacity query's own copy of the bound to move with it, and because those widths are unqualified
   on either route today - but the probe exists and I can prepare it as a separate change.
