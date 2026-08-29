**Level of the claim: operator.** The end-to-end figures at the end are an observation, not the
evidence.

## The observation

`w8_rowsplit_gemm_mma`'s `dequant_w` stages the weight tile one 16-bit code pair per lane. Each lane
decodes two codes and issues a 4-byte `store_vec`, so a row of the tile costs 32 four-byte shared
stores on the `BK == 64` schedules and 64 on the `BK == 128` one. A lane also cannot read its own
quantisation-group scale - its pair may belong to a group whose scale another lane loaded - so the
scales are broadcast with `__shfl_sync`, once per row pair on the `GROUPS == 2` branch and twice on
`GROUPS == 4`.

Giving each thread a whole eight-code run inside one group removes both: the thread reads its own
scale halfword, decodes eight codes out of one `uint2` and stores them with one 16-byte `store_vec`.
Three quarters of the shared stores, the loop trips around them, and all three shuffle sites go. Per
weight the arithmetic is unchanged - the same sign extension, the same multiply by the same group
scale, the same `__floats2bfloat162_rn`.

This is the projection half of the widening whose MoE half you merged as #106. I said there it was a
different contract and a different claim and would come separately; this is it. The implementation
exists ahead of this Issue only because #106 announced it and you merged that PR with the sentence in
it; nothing further goes in until you answer here.

`w8_rowsplit_gemm_mma.cuh` is instantiated by six public Op families - Linear, LinearAdd, LinearPair,
LinearSwiGLU, AttnInputProj and GdnInputProj. It is not a prefill-only change: the only `GROUPS == 4`
schedule, `MmaR64x16C48K128A1`, serves `k=5120, n=34816` at `t` in [41,48] and `k=5120, n=248320` at
`t` in [34,48] (`src/ops/linear/w8/w8_dispatch.cpp:30-39`), and the projections enter this kernel at
T=65 and T=97 - decode-shaped extents, reachable as a speculative batch under the one-to-eight active
requests the product targets.

## Measured effect

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, gcc 13.3, Ubuntu 24.04, Release,
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Base `1fc1cb76`. The figures were taken on `3a61ef3f`; the four commits since then touch `src/serve`,
the cache test oracle and the FP8 vocabulary GEMM, none of them this header or its consumers. Both
arms built in the same build directory, the same two binaries throughout. Each benchmark's default warmup (3 for `linear`, 5 for the others), 50
measured samples, L2 flushed for every one. Everything below was run at least twice: two independent
campaigns with separate builds of both arms, and three passes for the table just below. The one
exception is the 512-token output gate, a single run per arm. Ratios are master /
branch.

**The operator at the extents the product runs.** The shipped suites stop at T=1024 and prefill runs
at chunk 8192, so I measured the four W8 shapes a 35B prefill actually uses, at the extents it uses
them:

| n, k | T=1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| 12288, 2048 | 1.080-1.086 | 1.086-1.090 | 1.087 | 1.087-1.089 |
| 9216, 2048 | 1.093-1.103 | 1.090 | 1.088-1.089 | 1.088-1.090 |
| 2048, 4096 | 1.071-1.091 | 1.083 | 1.096-1.100 | 1.089-1.092 |
| 2048, 16384 | 1.089 | 1.118-1.121 | 1.096-1.099 | 1.093-1.095 |

Three independent passes per cell, so each cell is the range over its three.
**Median x1.090 over the 48 points, range x1.071 to x1.121.** The `w8_linear_add --production-only` sweep reports a higher
figure, x1.123 median over its 18 row-split rows at T=129-1024, but those are extents the model does
not execute; x1.090 is the number the end-to-end result rests on, and it agrees with the in-product
kernel measurement below.

**Where the routes hand the shape over.** Three dispatch tables put a boundary inside a sweep, and
the speedup starts at each one - which is the check that these sweeps measure what they claim to.
`k=5120` W8 linear, `--sweep 30:56`:

| n | T=30-33 | T=34-40 | T=41-48 | T=49-56 |
|---|---|---|---|---|
| 34 816 | x1.000 `small_t` | x0.989-1.000 `small_t` | **x1.075-1.100** `r64x16_c48_k128_a1` | **x1.057-1.065** `mma_r64_c128` |
| 248 320 | x1.000-1.004 `small_t` | **x1.149-1.187** `r64x16_c48_k128_a1` | **x1.085-1.184** same | **x1.074-1.084** `mma_r32_c64` |

`small_t` runs to T=40 at `n=34816` and only to T=33 at `n=248320`, and the ratio steps at those two
different extents accordingly. The projections do the same at their own boundaries, T=97 for
`gdn_input_proj` and the `attn_input_proj` companion and T=65 for the `attn_input_proj` target: below
them all three w8 rows sit at x1.000 on `SplitKMmaDirect`, and at the boundary they step to x1.062,
x1.105 and x1.045.

**Confinement.** The routed MoE prefill benchmark - #106's kernels, untouched here - gives x0.997 to
x1.000 across all nine codec/extent cells. At product scale, `nsys` kernel totals for one
33,031-token prefill at chunk 8192: `w8_rowsplit_gemm_mma` 491.85 -> 454.75 ms (**x1.082**, 25.8% of
GPU time), and across four profile pairs no other kernel group deviates from unity by more than
0.27%. The launch count of every one of the 67 kernels is identical between arms.

## Tradeoffs

**Registers**: across the 120 instantiations that match by name between builds, maximum 127 -> 117,
mean 102.4 -> 96.2, lower in 103, higher in 9, unchanged in 8. **Shared memory**: the same set of 22
values. **Spills**: none either side. **Workspace, resident memory, transfers, graph nodes**:
unchanged - no allocation added or removed, no boundary moved.

**Instruction stream**: of 2899 name-matched device functions, 120 differ and 2779 are identical in
both instruction count and per-opcode census; all 120 are instantiations of this kernel. Over them
the count falls 179,948 -> 161,238. The shuffle machinery leaves entirely - `SHFL` 1150 -> 0,
`WARPSYNC` 1360 -> 0, `ENDCOLLECTIVE` 390 -> 0 - and staging falls with it, `STS` 2884 -> 1724,
`LDS` 2934 -> 1514. What is identical body for body across all 120 is the part that carries the
claim: **`HMMA` 15120**, `LDSM` 12480, `LDGSTS` 1802, `FFMA` 4160, `STG` 1882, `BAR` 1302. The
tensor-core stream and everything feeding it are untouched; what gets cheaper is operand preparation.

**Numerical quality**: bit-for-bit identical, measured rather than argued. Two throwaway harnesses -
the old and new `dequant_w` side by side in one CTA over all 26 distinct `(BM, BK, THREADS)`
instantiations, every scale-cache residue, adversarial FP16 scale bit patterns and a sentinel in
every shared slot (zero mismatches, zero unwritten), and a whole-kernel A/B from two binaries over 33
cases covering both `Full` values, all three epilogues, both `BK`, ragged shapes and a mid-loop
scale-cache refill (983,163 bytes identical).

**Other execution paths**: one function in one header. The decode-shaped W8 kernels keep their own
staging; in the shipped suites their rows are the harness floor, which three repeat passes per arm
put at a p90 of x1.002 to x1.076 and a worst case of x1.164 - on rows this branch does not compile
differently. Those rows also carry the widest figures in the whole report: the T=1 decode row reads
x0.859 across arms, and one shape, `35b.dflash_feature`, reads x1.066 at T=1 and x0.957 at T=16.

`ctest -j1` on the rebased base: **94 tests, 0 failed, 1 skipped**, on master and on this branch. The
skip needs a 27B artifact this box does not have. On the old base this suite had one failure, the
one we reported as #105; `fd48e2fa` between the two bases rewrites that test's oracle and it now
passes.


**Re-taken on the current base.** Both arms were rebuilt on `1fc1cb76` after the rebase and the
whole grid re-measured, one more pass:

| n, k | T=1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| 12288, 2048 | 1.088 | 1.090 | 1.089 | 1.088 |
| 9216, 2048 | 1.103 | 1.090 | 1.089 | 1.089 |
| 2048, 4096 | 1.091 | 1.084 | 1.096 | 1.092 |
| 2048, 16384 | 1.089 | 1.121 | 1.097 | 1.093 |

Median **x1.090**, range x1.084 to x1.121 - the same median as the campaign above and the same top
of range. Three cells in the first row land 0.1 to 0.3% above their earlier per-pass range and two
land just below theirs, all of it inside the run-to-run envelope stated below.

## End-to-end observation

Arms alternate inside each round, one process per point, greedy, four rounds. Percent faster than
master.

| model | chunk | prompt tokens | prefill | decode | round spread |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 8192 | 8 515 | **+2.41%** | -0.01% | 0.34% |
| Qwen3.6-35B-A3B | 8192 | 33 031 | **+1.98%** | +0.01% | 0.28% |
| Qwen3.6-35B-A3B | 8192 | 65 882 | **+1.59%** | -0.00% | 0.17% |
| Qwen3.6-35B-A3B | **1024** | 33 031 | **+1.74%** | +0.01% | 0.11% |
| Qwen3.6-27B dense | 8192 | 33 031 | -0.04% | -0.05% | 0.42% |

1,024 is the product default and its row is the +1.74% line. At every 35B point the slowest candidate
run is faster than the fastest baseline run, in both campaigns. A full generation - 33,031 tokens in,
`--max-new 512`, stopping naturally at 164 tokens - is byte-identical between the arms. The dense 27B
row is a null result rather than a control: its W8 row-split tensors are the MTP and vision entries,
which a text-only greedy run with speculation off does not reach.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host, so the kernel-level figures come
  from `cuobjdump` rather than hardware counters.
- The 38 internal-linkage `w8_pair_gemm_concat.cu` instantiations cannot be matched by symbol between
  two builds, so they are covered by the `linear_pair` measurement and the output gate rather than by
  the register or instruction census.
- The `linear` suite, the `GROUPS == 4` sweeps and the routed-MoE control were run once per campaign,
  so they carry only the campaign-to-campaign spread.
- Speculation is off end to end, so the decode-shaped extents covered at operator level are not
  exercised in the product runs.
- Qwen3.8-27B NVFP4 is not measured; it takes no W8 row-split route.
- `docs/performance.md` publishes `qwen3_6_35b_a3b` prefill at a 1,024-token chunk with INT8 group-64
  KV, a path this change is on, so those figures would move. I have not re-run that harness.

## Questions

1. Is this within scope, and is an external implementation appropriate here? It is one commit in one
   file - 25 added, 35 removed - and I can open a pull request against `master` linked to this Issue,
   or leave the report and the numbers if you would rather implement it yourself.
2. Two merged changes already move the published `qwen3_6_35b_a3b` prefill figures in
   `docs/performance.md`, and this would be a third. Would you like the serve-corpus harness re-run
   and the document updated - with this, as a follow-up Issue, or not at all?
