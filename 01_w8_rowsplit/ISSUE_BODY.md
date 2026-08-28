**Level of the claim: operator.** The end-to-end figures at the bottom are an observation, not the
evidence; the claim is about one GEMM.

## The observation

`w8_rowsplit_gemm_mma`'s `dequant_w` stages the weight tile one 16-bit code pair per lane. Each
lane decodes two codes and issues a 4-byte `store_vec`, so a row of the tile costs 32 four-byte
shared stores on the `BK == 64` schedules and 64 on the `BK == 128` one. A lane also cannot read
its own quantisation-group scale: its pair may belong to a group whose scale another lane loaded,
so the scales are broadcast with `__shfl_sync` - once on the `GROUPS == 2` branch and twice on
`GROUPS == 4`.

Giving each thread a whole eight-code run inside one group removes both. The thread reads its own
scale halfword, decodes eight codes out of one `uint2` and stores them with a single 16-byte
`store_vec`: three quarters of the shared stores, the loop trips around them, and all three shuffle
sites go. Per weight the arithmetic is unchanged - the same sign extension, the same multiply by
the same group scale, the same `__floats2bfloat162_rn`.

This is the projection half of the widening whose MoE half you merged as #106. I said there that it
was a different contract and a different claim and that it would come separately; this is it,
rebased onto `master` at `3a61ef3f`, which already carries that half. `dev` is at `3a61ef3f` too,
so nothing unmerged touches this header. The implementation exists ahead of this Issue only because
#106 announced it and you merged that; nothing further goes in until you answer here.

`w8_rowsplit_gemm_mma.cuh` is instantiated by six public Op families - Linear, LinearAdd,
LinearPair, LinearSwiGLU, AttnInputProj and GdnInputProj - so all six are measured below. It is not
a prefill-only change: the only `GROUPS == 4` schedule, `MmaR64x16C48K128A1`, is selected at
`k=5120, n=34816` for `t` in [41,48] and at `k=5120, n=248320` for `t` in [34,48]
(`src/ops/linear/w8/w8_dispatch.cpp:30-39`). No shipped **benchmark** samples those extents - the
`linear` suite's continuous T list is `{1, 16, 128, 1024}` - so they are swept explicitly below.
The Op tests do reach them, which matters for the numerical argument and is covered there.

## Measured effect

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, gcc 13.3, Ubuntu 24.04, Release,
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Baseline is `3a61ef3f` built in the same build directory as the
candidate; both arms are the same two binaries throughout every table here. Each benchmark's own
default warmup (3 for `linear`, 5 for the others) and 50 measured cold-L2 samples through
`bench::measure_cold_launch`.

```bash
./build/bench/ninfer_w8_linear_add_bench --production-only --repeat 50
./build/bench/ninfer_w8_linear_swiglu_bench --production-only --repeat 50
./build/bench/ninfer_linear_bench --suite all --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 34816  --k 5120 --sweep 30:56 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 248320 --k 5120 --sweep 30:56 --repeat 50
./build/bench/ninfer_linear_pair_bench --tokens 64,128,192,193,256,384,512,768,1024 --repeat 50
./build/bench/ninfer_gdn_input_proj_bench --repeat 50
./build/bench/ninfer_attn_input_proj_bench --repeat 50
./build/bench/ninfer_sparse_moe_bench --codec all --tokens 1024 --cache cold --execution eager --repeat 50
./build/bench/ninfer_sparse_moe_bench --codec all --tokens 4096 --cache cold --execution eager --repeat 50
./build/bench/ninfer_sparse_moe_bench --codec all --tokens 8192 --cache cold --execution eager --repeat 50
```

The MoE benchmark takes `--execution eager` because prefill is not graph captured: the only capture
site is `src/core/decode_graph.cpp` and the prefill chunk loop is outside it.

The kernel profile and the end-to-end runs, one process per point:

```bash
nsys profile -t cuda --cuda-graph-trace=node -o p --force-overwrite=true \
  ./build/apps/ninfer <artifact> --messages <prompt> --max-new 1 --greedy --no-thinking \
  --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o k p.nsys-rep

./build/apps/ninfer <artifact> --messages <prompt> --max-new 32 --greedy --no-thinking \
  --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16
```

### The changed kernel, and the benchmark's own floor

`w8_linear_add --production-only` covers four kernels and only one of them is this one, so the rows
are grouped by the kernel each shape actually reaches. All ratios are master / branch, so above 1.0
is faster:

| kernel | rows | T | median | best | worst |
|---|---:|---|---:|---:|---:|
| `w8_rowsplit_gemm_mma` | 18 | 129-1024 | **x1.123** | x1.211 | x1.018 |
| `w8_small_t_mma` (`splitk8`) | 21 | 2-48 | x1.000 | x1.001 | x0.999 |
| `medium_splitk` | 10 | 63-128 | x1.000 | x1.043 | x0.960 |
| decode | 1 | 1 | x0.857 | - | - |

Three of those four kernels are not compiled differently by this branch, so their rows are the
harness floor rather than a control for the change, and it is not a tight floor. Rather than assert
that, I measured it: three independent passes of the same benchmark against the same binary, per
arm, reported as max/min across the three passes for each row.

| benchmark | arm | rows | median pass-to-pass spread | p90 | worst |
|---|---|---:|---:|---:|---:|
| `w8_linear_add --production-only` | master | 50 | x1.000 | x1.035 | x1.161 |
| `w8_linear_add --production-only` | branch | 50 | x1.000 | x1.026 | x1.077 |
| `w8_linear_swiglu --production-only` | master | 39 | x1.000 | x1.031 | x1.100 |
| `w8_linear_swiglu --production-only` | branch | 39 | x1.000 | x1.043 | x1.100 |

The worst row of the master arm is `linear_add.w8.decode.r16.residual` at T=1, at x1.161 against
itself - the same row that reads x0.857, that is x1.167 the other way, in the cross-arm table above.
The worst two `swiglu` rows are `splitk.mma.pair.exact_t` at T=3 and T=11, at x1.100 and x1.091
against themselves, and those are the same two rows that read x0.909 and x0.918 below. Every figure
below unity anywhere in this report sits on a row whose own repeat spread is comparable or wider,
and every one of them is on a kernel this branch does not compile differently. Read the changed
kernel's x1.018 worst case against that floor, not against x1.000.

### The other five Op families

| benchmark | rows that move by >1% | median of those | best | worst |
|---|---:|---:|---:|---:|
| `w8_linear_swiglu --production-only` | 20 of 39 | **x1.099** | x1.143 | x0.909 |
| `linear --suite all` | 20 of 68 | **x1.089** | x1.171 | x0.965 |

The `linear` suite carries Q4, Q5, Q6, BF16, NVFP4, FP8 and W8 rows. Nineteen of the 20 movers are
W8 rows. Two rows move down: a Q5 `vision_fc2` row at T=1024 at x0.985, which is the one non-W8
mover, and a W8 `27b.mtp_attention` row at T=1 at x0.965. Of the 33 W8 rows in the suite, 14 sit
within one percent of unity. The two `linear_swiglu` rows below unity are `splitk.mma.pair.exact_t`
at T=3 and T=11 - the two rows the repeat table above puts at x1.100 and x1.091 against themselves.

`linear_pair` puts a route boundary on display: `kK2048Routes`
(`src/ops/linear_pair/w8/w8_pair_plan.cpp:30`) first selects a `ConcatMma` schedule that
instantiates this header at T=193.

| T | 64 | 128 | 192 | 193 | 256 | 384 | 512 | 768 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| speedup | x1.000 | x1.000 | x1.007 | **x1.136** | **x1.133** | **x1.200** | **x1.118** | x1.042 | x1.068 |

### The GROUPS == 4 schedule

Swept at `k=5120` across every route boundary in the interval. The speedup appears exactly where
the dispatch table hands the shape to this kernel and vanishes below it, which is the check that
the sweep measures what it claims to.

| n | T=30-33 | T=34-40 | T=41-48 | T=49-56 |
|---|---|---|---|---|
| 34 816 | x1.000-1.013 `small_t` | x1.000-1.012 `small_t` | **x1.085-1.100** `r64x16_c48_k128_a1` | **x1.058** `mma_r64_c128` |
| 248 320 | x0.992-0.998 `small_t` | **x1.146-1.187** `r64x16_c48_k128_a1` | **x1.081-1.185** same | **x1.066-1.073** `mma_r32_c64` |

The two rows part company exactly where the dispatch table says they should: `small_t` runs to T=40
at `n=34816` and only to T=33 at `n=248320`, and above T=48 the two shapes go to different wide
schedules of the same kernel - `MmaR64C128` and `MmaR32C64`. Three distinct schedules of this
kernel are therefore visible in one sweep, all moving, with the untouched `small_t` route flat
underneath them.

### The projections

The two projection benchmarks measure one operator in several weight formats, so the change and the
benchmark's own floor sit side by side:

| operator | format | T=64 | 128 | 256 | 512 | 1024 |
|---|---|---:|---:|---:|---:|---:|
| `gdn_input_proj` | **w8** | x1.000 | **x1.065** | **x1.109** | **x1.097** | **x1.073** |
| `attn_input_proj` | **w8-qkv** | x1.000 | **x1.105** | **x1.159** | **x1.102** | **x1.099** |
| `attn_input_proj` | **w8-qgkv** | x0.991 | **x1.044** | **x1.067** | **x1.053** | **x1.093** |
| both, 8 format rows | bf16, fp8, nvfp4, q4q5 | x1.000 | x1.000 | x1.000-1.031 | x0.993-1.016 | x0.989-1.000 |

The last row collapses eight rows - four weight formats on each of the two operators. From T=64 up
the untouched formats are flat with one exception, nvfp4 `gdn_input_proj` at T=256 at x1.031. Below
T=64 the same benchmarks, whose default sweep starts at T=1, put those untouched formats between
x0.967 and x1.048, which is their short-extent floor.

### Confinement in the sibling's kernels

The routed MoE prefill benchmark. This branch touches none of those kernels, and none of them moves:

| codec | 1024 | 4096 | 8192 |
|---|---:|---:|---:|
| q4-q5 | x0.997 | x0.999 | x1.000 |
| q4-q6 | x0.997 | x1.000 | x0.999 |
| w8-w8 | x1.000 | x0.999 | x0.998 |

## The same question at product scale

The operator benchmarks and the end-to-end runs are two different things and neither explains the
other, so here is the step between them: `nsys` kernel totals for one 33,031-token prefill at chunk
8192, one profiled run per arm, same command. These are deterministic per-kernel sums over a fixed
launch count rather than a repeat statistic, so the precision below is the profiler's, not a
confidence interval.

| kernel group | master | branch | speedup | share of master GPU time |
|---|---:|---:|---:|---:|
| `w8_rowsplit_gemm_mma` (3 forms) | 488.8 ms | 452.4 ms | **x1.0805** | 25.9% |
| `causal_attention_prompt_bf16` | 493.7 ms | 493.8 ms | x0.9999 | 26.1% |
| `sparse_moe_prefill_q4_gate_up` | 360.1 ms | 360.0 ms | x1.0001 | 19.0% |
| `sparse_moe_prefill_qx_down` | 194.8 ms | 194.8 ms | x1.0001 | 10.3% |
| gated delta net (4 kernels) | 109.8 ms | 109.8 ms | x1.0003 | 5.8% |
| `sparse_moe_prefill_gather` | 28.8 ms | 28.8 ms | x0.9997 | 1.5% |
| **all kernels** | **1890.8 ms** | **1854.5 ms** | **x1.0196** | 100% |

Three things worth reading off this. The in-product speedup on the changed kernel is x1.081, below
the x1.12 the cold-L2 benchmark reports, which is what a warm L2 and a mix including the narrow
schedules should do. Every other kernel group sits inside 0.03% of unity, which is the confinement
claim measured at product scale rather than argued from the diff. And the total, +1.96%, sits
against the +1.97% the end-to-end table reports at the same point - close enough that the three
levels tell the same story, not so close that I would call it an identity.

## Tradeoffs

**Registers.** The binary carries 158 instantiations of this kernel; 120 can be matched by name
between the two builds and 38 cannot, because the instantiations from
`src/ops/linear_pair/w8/w8_pair_gemm_concat.cu` have internal linkage and their symbols carry a
per-compilation module id. Across the 120: maximum 127 -> 117, mean 102.4 -> 96.2, lower in 103 and
higher in 9. **Shared memory**: the same set of 22 distinct values on both sides. **Spills**: none
on either side, stack unchanged. The 38 unmatched are exactly the instantiations behind the
`linear_pair` row above, which is where their behaviour is measured instead.

**Workspace, resident memory, transfers, graph nodes.** Unchanged. No allocation is added or
removed, no arena or workspace boundary moves, no node appears or disappears.

**Instruction stream.** `cuobjdump -sass` on the two `ninfer` binaries, compared body by body. Each
binary carries 2931 device functions; 2893 match by name and 38 do not, for the linkage reason
above. Of the 2893 matched, **120 differ and 2773 are identical in both instruction count and
per-opcode census**, and every one of the 120 is an instantiation of this kernel. Counts fall
between 1.2% and 32.4%, median 4.9%; most instantiations are narrow schedules where the staging
loop is a small share of the body.

| instantiation | instructions | STS | LDS | I2F | HMMA |
|---|---:|---:|---:|---:|---:|
| deepest reduction | 1976 -> 1336 (-32.4%) | 72 -> 37 | 61 -> 11 | 80 -> 10 | 160 -> 160 |
| most instructions | 2616 -> 1984 (-24.2%) | 40 -> 5 | 60 -> 10 | 80 -> 10 | 160 -> 160 |
| BN=128 form | 1920 -> 1448 (-24.6%) | 30 -> 5 | 45 -> 10 | 60 -> 10 | 160 -> 160 |

Summed across all 120 changed bodies: `SHFL` **1150 -> 0**, `STS` 2884 -> 1724, `LDS` 2934 -> 1514,
`I2F` 3560 -> 1240, `LOP3` 18003 -> 12982, `IMAD` 16476 -> 15493, and `HMMA` 15120 -> 15120 with
**zero bodies in which the HMMA count moves**. One counted opcode rises: `F2FP`, 3994 -> 4694.
These are static counts in differently unrolled loop bodies rather than per-weight rates, so the one
that carries the claim is `HMMA`: the arithmetic the tensor cores do is untouched, and what gets
cheaper is the preparation of their operands.

**Numerical quality.** The intent is bit-for-bit identity, and per weight the arithmetic is the
expression it replaces. Only the route by which the scale reaches the thread changes: for
`GROUPS == 2` the old code loaded a `uint32` at `scale_pair_offset` and extracted
`>> ((gg & 1) * 16)`; for `GROUPS == 4` it shuffled two `uint32` from `+0` and `+4`. Both resolve to
the halfword at `scale_pair_offset + gg * 2`, which is what the thread now reads directly, and the
highest byte touched is still 16, which is `SCALE_CACHE_BYTES`. The 16-byte store lands on the same
eight shared slots in the same order as the four 4-byte stores it replaces:
`w8g32_swz64(row, col) = (((col >> 3) ^ (row & 7)) << 3) | (col & 7)`, so at `col = 8c` the low
three bits vanish, the offset is a multiple of eight elements and therefore 16-byte aligned, and
`c ^ (row & 7)` is a bijection on a row's chunks. Removing the shuffles is safe because they sat in
a warp-uniform loop: the old trip count depended only on `warp`, so all 32 lanes were converged.

The shipped Op tests qualify the routes against an independent oracle but compare with tolerances
rather than bits, so they establish that the change is correct, not that it is identical.
`tests/ops/linear/test_w8_a16.cpp` runs against `cpu_linear_gemm_fp64`, and its shape lists step
across every route boundary this change touches. `kN248320K5120` carries `a16(33), a16(34), a16(48),
a16(49), a16(64), a16(65)`, which straddles all three of that shape's boundaries; `kN34816K5120`
carries `a16(40), a16(41), a16(48), a16(49)`, which straddles both of its own. Between them they
qualify the `MmaR64x16C48K128A1` window (`BK == 128`, `GROUPS == 4`) where two of the three removed
shuffle sites lived, and the wide `GROUPS == 2` schedules on either side of it. `tests/ops/linear_add`, `linear_pair`,
`linear_swiglu`, `test_attn_input_proj.cpp` and `test_gdn_input_proj.cpp` cover the other five Op
families. What establishes bit-identity on top of that is the `HMMA` census above and greedy output
byte-identical to `master` at every measured point.

**Other execution paths.** The change is one function in one header. The decode-shaped W8 kernels
(`w8_small_t_mma`, `medium_splitk`, the SIMT decode routes) keep their own staging and are the
unchanged rows in the tables above.

```bash
cd build && ctest -j1
```

92 pass, 1 skipped, 1 failed - on this branch and on `master` built in the same directory in the
same campaign. The skip is `27b_load_plan`, which needs both real 27B artifacts and only one is on
this box. The failure is `ninfer_qwen3_6_27b_prefix_real_test`, with the same message on both arms;
it is #105 and still open.

## End-to-end observation

Not offered as isolation, only as a check that the operator result survives in the product. Arms
alternate inside each round, every point its own process, greedy, four rounds each. Prefill and
decode columns are percent faster than master, higher is better; spread is the largest
round-to-round spread among the arms in that row.

| model | chunk | prompt tokens | rounds | prefill | decode | spread |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 8192 | 8 515 | 4 | **+2.34%** | +0.03% | 0.49% |
| Qwen3.6-35B-A3B | 8192 | 33 031 | 4 | **+1.97%** | -0.01% | 0.50% |
| Qwen3.6-35B-A3B | 8192 | 65 882 | 4 | **+1.56%** | +0.01% | 0.29% |
| Qwen3.6-35B-A3B | 1024 | 33 031 | 4 | **+1.73%** | +0.01% | 0.16% |
| Qwen3.6-27B dense | 8192 | 33 031 | 4 | -0.05% | -0.01% | 0.44% |

At every 35B point the slowest of the four candidate runs is faster than the fastest of the four
baseline runs, so the separation is not an artifact of round ordering. All 20 generations are
byte-identical to `master`. The gain falls with prompt length because attention's share of prefill
grows while the GEMM's falls, and the change lives only in the GEMM.

The dense 27B row is close to a null result rather than a control: its W8 row-split tensors are the
MTP and vision entries, and a text-only greedy run with speculation off does not reach them.

## Scope and ownership

One function in `src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh`, 22 added and 35 removed lines. No
public header, no CLI or serving surface, no artifact contract, no workspace or graph boundary, no
new environment variable, no `getenv`, no `thread_local`, no module-global device state.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host, so occupancy and memory-pipe
  counters could not be collected. The register and shared-memory figures come from
  `cuobjdump --dump-resource-usage` and the instruction counts from `cuobjdump -sass` instead.
- The 38 concat instantiations are covered by the `linear_pair` measurement and the greedy-output
  gate, not by the instruction or register census.
- Speculation is off in every run here, so the MTP leaves are not exercised.
- Qwen3.8-27B NVFP4 is not measured; it takes no W8 row-split route.
- The `nsys` profile is one run per arm rather than a repeated measurement.
- `docs/performance.md` publishes prefill throughput for `qwen3_6_35b_a3b` measured at a 1,024-token
  prefill chunk with INT8 group-64 KV, a path this change is on, so those figures would move. I have
  not re-run that harness - it uses prefix reuse, CUDA Graphs and five seeds through the serve
  corpus - and I am not proposing edits to the document here. See the question below.

## Questions

1. Is this within scope, and is an external implementation appropriate here? It is implemented and
   measured as one commit in one file and I can open a pull request against `master` linked to this
   Issue, or leave the report and the numbers if you would rather implement it yourself.
2. Two merged changes now move the published `qwen3_6_35b_a3b` prefill figures in
   `docs/performance.md`, and this would be a third. Would you like the serve-corpus harness re-run
   and the document updated - as part of this, as a follow-up Issue, or not at all?
