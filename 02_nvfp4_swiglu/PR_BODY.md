Implements #ISSUE.

Scope as confirmed in #ISSUE: …

`resolve_route` took the fused TMA SwiGLU at exactly `T == 1024`, although the kernel accepts any
`T % 256 == 0`.
Every wider prefill chunk therefore fell into `LinearW4A4Post`, which materialises the `34816 x T`
bf16 projection into the arena and reads it back through a separate `silu_mul`.

The claim is at operator level. The end-to-end section is confirmation, not the evidence.

## The design

```cpp
-    if (tokens == kPrimaryT) { return Nvfp4LinearSwiGluRoute::TmaFusedW4A4; }
+    if (tokens >= kPrimaryT && (tokens % kTmaBlockM) == 0) {
+        return Nvfp4LinearSwiGluRoute::TmaFusedW4A4;
+    }
```

Two clauses, both already true of the kernel: `kTmaBlockM` is the block extent the TMA schedule
tiles `T` by, and `kPrimaryT` remains the lower bound because it is the width the fused route is
already registered and oracle-qualified at - not because it is where the fused path starts winning,
which it is not; `The lower bound` below measures how much that leaves.
The capacity query is widened to match - it sizes the fused path for the largest accepted width in
the interval rather than for `kPrimaryT` alone. It `std::max`es rather than assigns, which matters
for the term above it, not below: the old `maximum = fused_workspace_bytes(kPrimaryT)` clobbered
the small-T fused term from the `min_tokens <= 48` branch. The materialising `last_baseline` term
below already maxed and is untouched.

Nothing else moves: no kernel is edited, and the compiled result agrees - `cuobjdump -sass` over the
two binaries finds **0 of 2899 comparable device functions changed** (2937 enumerated, 38 unmatched
by name because they are concat symbols with internal linkage), registers, shared memory and spills
identical everywhere. Only which kernel a given width reaches changes.

## Affected behaviour and contract

`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`, +10/-4, one file, one commit.

**The workspace capacity is contract, so it was enumerated rather than argued.** The function was
called on both arms over a grid of 4560 `(min_tokens, max_tokens)` intervals:

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
the materialising path becomes unreachable and the whole projection leaves: 594,018,304 → 23,592,960 bytes at 8192, and
34816 x 8192 x 2 = 570,425,344 exactly.

The mechanism behind the zero is worth stating, so the sweep reads as a proof rather than as brute
force: `baseline_workspace_bytes` exceeds `fused_workspace_bytes` at the same width by roughly 25x
(594,018,304 against 23,592,960 at 8192). So whenever the interval still contains a width that
routes to `LinearW4A4Post`, the baseline term is the maximum and the widened fused term cannot
surface. The only intervals where the total moves materially are the eleven with no such width left.

**The engine does not request a collapsed interval today**, and the measured peak settles that
without reading further code: it is identical on both arms, 152.57 MiB at `--prefill-chunk 1024` and
1.19 GiB at 8192. Had the interval been collapsed, 544 MiB would have left the peak. So the win is
traffic, not capacity, and I would rather say that than quote the 544 MiB.

**The two routes are not bit-identical and cannot be.** The fused epilogue holds the f32 accumulator
and rounds once; the materialising one rounds gate and up to bf16 and multiplies the rounded values.
This does not introduce that difference, it moves which widths see it: stock already answers
differently at 1024 and at 8192, and after this both answer the way 1024 does. Numerics are below.

No public header, no CLI or serving surface, no artifact contract, no graph node, no new environment
variable, no `getenv`, no `thread_local`, no module-global device state.

## Verification

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, gcc 13.3, Ubuntu 24.04, Release,
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Base `3a61ef3f`, both arms built in the same build directory, the
same two binaries throughout. Artifact `qwen3.8-27b/nvfp4`. Benchmark default warmup 5, 50 measured
samples. Ratios are master / branch, so above 1.000 is faster.

```bash
./build/bench/ninfer_nvfp4_linear_swiglu_bench --policy a4 \
  --t-sweep 256,512,768,1024,1280,2048,4096,8192 --repeat 50
./build/bench/ninfer_nvfp4_linear_swiglu_bench --policy a4 \
  --t-sweep 1024,1025,1152,1279,1280,1281,1408,1536,1792,2048,2304,2560 --repeat 50

# the numerics table below. The kA4Cases extension is NOT in this commit - it was applied on top
# of each arm to take those rows, because whether it belongs in the change is Question 2 in the
# Issue. Extend kA4Cases to {5, 48, 49, 128, 1024, 1280, 2048, 4096} to reproduce them.
NINFER_OP_REPORT_STATS=1 ./build/tests/ninfer_linear_swiglu_nvfp4_test

# end-to-end rounds, one process per point, arms alternating inside each round
./build/apps/ninfer qwen3_8_27b_nvfp4.ninfer --messages niah_16k.json --max-new 32 --greedy \
  --no-thinking --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16
./build/apps/ninfer qwen3_8_27b_nvfp4.ninfer --messages niah_32k.json --max-new 32 --greedy \
  --no-thinking --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16
./build/apps/ninfer qwen3_8_27b_nvfp4.ninfer --messages niah_16k.json --max-new 32 --greedy \
  --no-thinking --max-context 131072 --prefill-chunk 1024 --kv-dtype bf16
# the output gate
./build/apps/ninfer qwen3_8_27b_nvfp4.ninfer --messages niah_32k.json --max-new 512 --greedy \
  --no-thinking --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16

cd build && ctest -j1
```

### The operator

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

The `1.000` cells are not a noise floor: below 1280 both arms resolve to the same route and run the
same kernel, so they are a built-in control that nothing the table already covered has moved.
Throughput over the same points goes 874 → 874 TFLOP/s at 1024 and 845 → 946 at 8192.

### The predicate, swept on its own

| T | 1024 | 1025 | 1152 | 1279 | **1280** | 1281 | 1408 | **1536** | **1792** | **2048** | **2304** | **2560** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T % 256` | 0 | ≠0 | ≠0 | ≠0 | **0** | ≠0 | ≠0 | **0** | **0** | **0** | **0** | **0** |
| speedup | 0.995 | 1.000 | 1.000 | 1.000 | **1.053** | 0.997 | 1.000 | **1.065** | **1.063** | **1.079** | **1.084** | **1.087** |

The gain appears at every accepted width and at no other, including across the adjacent triple
1279 / 1280 / 1281. The two cells that are not exactly unity - 1024 at 0.995, already fused on both arms, and 1281 at
0.997, not a multiple of 256 - are the run-to-run floor on a route neither arm changes, and 2048 reads 1.079 here against
1.091 in the sweep above, which is that same floor from the other side.

On `master` the materialising path at 1025 costs 598.0 µs against 417.8 for the fused path at 1024,
so the table has a sawtooth at every 256-wide step above 1024. This removes it at the accepted
widths and leaves it in between.


**Re-taken on the current base.** Master moved to `1fc1cb76` while this was in review; both arms
were rebuilt there and both sweeps re-run in one more pass. The operator reads 1.000 / 1.006 / 1.000
/ 1.000 at 256 / 512 / 768 / 1024, then 1.053 at 1280, 1.091 at 2048, 1.130 at 4096 and 1.122 at
8192. The predicate sweep reads 1.000 at every width that is not a multiple of 256, and 1.051 /
1.065 / 1.067 / 1.080 / 1.084 / 1.090 at 1280 / 1536 / 1792 / 2048 / 2304 / 2560. Every cell is
inside the +-1.2% envelope; the one that moved most is 512, which reads 1.006 in this pass and 1.000
in the table above on a width neither arm re-routes, so that is the floor measuring itself.

### Numerics

`tests/ops/linear_swiglu/test_nvfp4.cpp` qualifies **both** routes against the FP64 oracle today -
`kA4Cases{5, 48, 49, 128, 1024}` puts 49 and 128 on the materialising path and 1024 on the fused one
- and this change moves neither, so its verdicts are unchanged. It does not cover 1280 and up, which
is exactly what does change route. The same test with that list extended, both arms, criterion
`rel_L2 <= 1.6e-1` with the gross ratio under 1.0:

| T | route change | master rel_L2 | branch rel_L2 | master gross | branch gross |
|---:|---|---:|---:|---:|---:|
| 5, 48, 49, 128, 1024 | none | 0.111 … 0.0527 | **identical** | 0.639 … 0.373 | **identical** |
| 1280 | → fused | 0.05159 | **0.05126** | 0.336 | **0.307** |
| 2048 | → fused | 0.05011 | **0.04969** | 0.286 | **0.261** |
| 4096 | → fused | 0.052342 | **0.052338** | 0.286 | **0.326** |

Both arms pass at every width. Every width whose route is unchanged reproduces to the last printed
digit, which is the confinement claim at the numerical level rather than the timing level. At the
re-routed widths the fused path is equal or very slightly closer on relative-L2 and **worse on the
gross tail at 4096**, 0.326 against 0.286 - so it is not strictly more accurate, and I am not
claiming that. For scale, the shipped widths reach 0.639 of their own gross limit against 0.326 here - noting the
limit is width-dependent (0.567 at T=5, 1.266 at 4096), so these are fractions of different
absolute bounds.

The comment above that criterion (`tests/ops/linear_swiglu/linear_swiglu_test_common.cpp:28`) is the
frame:

> The criterion belongs to the activation-compute profile, not the weight storage format or a
> private materialized/fused implementation.

### The lower bound

`kPrimaryT` is where the fused route is registered today and I kept it, but I can no longer
say it is where the fused path starts winning, because it is not. A probe that lowers only the
route predicate to `kTmaBlockM`, three runs (`median_us` over 50 after 5 warmup,
materialising against fused):

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

### Resources

No kernel is edited, so registers and shared memory cannot move, and `cuobjdump --dump-resource-usage`
over the two binaries confirms it: every instantiation of every kernel identical, no spills either
side. Workspace peak identical on both arms at both chunk widths (152.57 MiB at 1024, 1.19 GiB at
8192). The capacity query never asks for more over 4560 intervals, as above. No allocation is added
or removed, no transfer changes, no graph node appears or disappears. There is no fusion and no
prefetch here - the change routes to an existing fused kernel - so there is no producer-consumer or
Graph step to measure.

### Suite

`ctest -j1` on the rebased base: **94 tests, 0 failed, 1 skipped**, on master and on this branch.
The skip needs a 27B artifact this box does not have. On the old base this suite had one failure,
`ninfer_qwen3_6_27b_prefix_real_test`, which we reported as #105; `fd48e2fa` between the two bases
rewrites that test's oracle, so it now passes.

### End to end

Confirmation only. Arms alternating inside each round, one process per point, greedy, four rounds.

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
small numeric differences at the argmax, and these routes are not required to agree.

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
- The oracle sweep stops at 4096. 8192 went through the benchmark and the product but not the FP64
  oracle.
- The output gate is one prompt, one run per arm at each chunk width.
- Only `qwen3.8-27b/nvfp4` is measured. `qwen3.6-27b/nvfp4` takes the same route and has strictly
  greater exposure - it binds an NVFP4 `mlp/gate_up {34816,5120}` on every text layer, where the
  measured artifact is NVFP4 only below layer 56 - and I did not measure it.
- I have not checked whether the same registration gap exists on the FP8 SwiGLU route.
- No `nsys` kernel attribution for this change; the operator sweep and the route table carry the
  claim instead.
