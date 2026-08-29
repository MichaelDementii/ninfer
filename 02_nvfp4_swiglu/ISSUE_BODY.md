**Level of the claim: operator.** The end-to-end figures at the end are an observation, not the
evidence.

## The observation

`resolve_route` in `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` takes the fused TMA
SwiGLU at exactly `T == 1024`:

```cpp
if (tokens == kPrimaryT) { return Nvfp4LinearSwiGluRoute::TmaFusedW4A4; }
return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
```

The kernel accepts any `T % 256 == 0`, and `nvfp4_linear_swiglu_workspace_capacity_bytes` beside it
is already written in terms of that constraint. So every prefill chunk wider than 1024 falls into
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
1279 / 1280 / 1281. The two off-multiple cells that are not exactly unity, 1024 at 0.995 and 1281 at
0.997, are the run-to-run floor on a route neither arm changes; 2048 reads 1.079 in this sweep and
1.091 in the one above, which is the same floor seen from the other side.

Worth noting for its own sake: on `master` the materialising path at 1025 costs 598.0 µs against
417.8 for the fused path at 1024, so the current table has a sawtooth at every 256-wide step above
1024. This change removes it at the accepted widths and leaves it in between.

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

952 of the reductions are 72,512 bytes - one token of the materialised projection plus alignment -
because `last_baseline` steps down when `max_tokens` itself now resolves to the fused route. The
other 11 are intervals collapsed to a single accepted width, where the materialising path becomes
unreachable and the whole projection leaves: at 8192 that is 594,018,304 → 23,592,960 bytes, and
34816 x 8192 x 2 = 570,425,344 exactly.

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
| 4096 | → fused | 0.05234 | 0.05234 | 0.286 | **0.326** |

Both arms pass the A4 criterion at every width. Two honest readings: every width whose route is
unchanged reproduces to the last printed digit, which is the confinement claim at the numerical
level; and at the re-routed widths the fused path is equal or very slightly closer on relative-L2
but **worse on the gross tail at 4096**, 0.326 against 0.286. So it is not "strictly more accurate".
For scale, the widths the project already ships reach 0.639 of the same gross limit, so the new
route stays well inside the envelope already accepted.

The comment above that criterion is the frame I am working from, and it is yours:

> The criterion belongs to the activation-compute profile, not the weight storage format or a
> private materialized/fused implementation.

**Other execution paths.** The diff is one predicate and one capacity branch in a host-side route
table. No kernel is edited.

## End-to-end observation

Arms alternating inside each round, one process per point, greedy, four rounds. Percent faster than
master.

| prompt tokens | chunk | prefill | decode | round spread |
|---:|---:|---:|---:|---:|
| 33 031 | 8192 | **+2.38%** | -0.01% | 0.91% |
| 65 882 | 8192 | **+1.96%** | -0.05% | 0.70% |
| 33 031 | **1024** | **+0.01%** | +0.01% | 0.32% |

The 1024 row is the one I would look at first. You wrote when you closed #96 that the 1024 default
is deliberate and that the larger-chunk gain is not free; this change does not ask you to move that
default, and the default path measures as untouched rather than being argued to be. The two 8192
rows are for an operator who has already chosen a wider chunk.

A 512-token generation on the 33,031-token prompt, stopping naturally at 203 bytes, is byte-identical
between the arms at **both** chunk widths. That is agreement, not proof: greedy decoding absorbs
small numeric differences at the argmax, and the routes are not required to agree.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
- The oracle sweep stops at 4096; 8192 was not run through the FP64 oracle, only through the
  benchmark and the product.
- The output gate is one prompt and one run per arm at each chunk width.
- Only `qwen3.8-27b/nvfp4` is measured. No other registered artifact takes this route.
- I have not looked at whether the same registration gap exists on the FP8 SwiGLU route.

## Questions

1. Is this within scope, and is an external implementation appropriate? It is one commit in one
   file, +10/-4, and I can open a pull request linked to this Issue, or leave the report.
2. Should extending `kA4Cases` in `tests/ops/linear_swiglu/test_nvfp4.cpp` to cover the widths this
   re-routes be part of the change? The repo asks for relevant shapes and execution routes to be
   exercised, and those widths are currently unqualified on either route.
