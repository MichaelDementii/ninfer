Closes #ISSUE.

`w8_rowsplit_gemm_mma`'s `dequant_w` staged the weight tile one 16-bit code pair per lane: each
lane decoded two codes and issued a 4-byte `store_vec`, so a row of the tile cost 32 shared stores.
A lane also could not read its own quantisation-group scale, because its pair might belong to a
group whose scale another lane had loaded, so the scales were broadcast with `__shfl_sync` - once
on the `GROUPS == 2` branch and twice on `GROUPS == 4`.

Each thread now owns a whole eight-code run inside one group: it reads its own scale halfword,
decodes eight codes out of one `uint2` and stores them with a single 16-byte `store_vec`. Per
weight the arithmetic is unchanged, so what this changes is how operands are prepared, not what is
computed.

This is the projection half of the widening whose MoE half you merged as #106. I said there it was
a different contract and a different claim and would come separately; this is it, on `master` at
`3a61ef3f`, which already carries that half. `dev` is at the same commit.

## The design and why it fits

The loop is now indexed by thread over eight-code chunks - the same shape `stage_x` and `stage_w`
already use a few lines above it in the same kernel - rather than by row pair over lanes. Three
consequences, in the order they matter:

- **A thread owns a chunk, so it owns the scale.** The three `shfl_sync` call sites go, and with
  them the reason the loop had to be structured around `lane >> 4` and `lane & 15`.
- **Eight decoded codes are sixteen contiguous bytes in shared memory**, so the four 4-byte stores
  become one 16-byte store. `w8g32_swz64(row, col) = (((col >> 3) ^ (row & 7)) << 3) | (col & 7)`,
  and at `col = 8c` the low three bits vanish: the offset is a multiple of eight elements, hence
  16-byte aligned, and `c ^ (row & 7)` is a bijection on a row's chunks, so the eight elements land
  in the same eight slots in the same order as before.
- **One invariant stops being structural and becomes an assumption**, so it is stated:

```cpp
constexpr int CodesPerChunk = 8;
constexpr int ChunksPerRow  = BK / CodesPerChunk;
// One scale covers a whole chunk, so a chunk may not straddle two quantisation groups.
static_assert((32 % CodesPerChunk) == 0, "an eight-code chunk must lie inside one W8G32 group");
```

The old loop made straddling impossible by construction; with a thread-indexed loop it holds
because eight divides the group width of 32.

## Affected behaviour and ownership

One function in `src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh`, +24/-33. No public header, no CLI or
serving surface, no artifact contract, no workspace or arena boundary, no graph node, no new
environment variable, no module-global or thread-local state. Numerically the intent is bit-for-bit
identity, which the verification below is built to establish.

The header is instantiated by six public Op families - Linear, LinearAdd, LinearPair, LinearSwiGLU,
AttnInputProj and GdnInputProj - so all six are measured. It is not a prefill-only change: the only
`GROUPS == 4` schedule, `MmaR64x16C48K128A1`, is selected at `k=5120, n=34816` for `t` in [41,48]
and at `k=5120, n=248320` for `t` in [34,48] (`src/ops/linear/w8/w8_dispatch.cpp:31-38`). Nothing
in the shipped benchmark suites samples those extents, so they are swept explicitly rather than
identified by reading the dispatch table.

## Verification

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, gcc 13.3, Ubuntu 24.04, Release,
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Baseline `3a61ef3f` built in the same build directory as the
candidate; both arms are the same two binaries throughout every table below. `--repeat 50`, cold L2
through `bench::measure_cold_launch`.

```bash
./build/bench/ninfer_w8_linear_add_bench --production-only --repeat 50
./build/bench/ninfer_w8_linear_swiglu_bench --production-only --repeat 50
./build/bench/ninfer_linear_bench --suite all --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 34816  --k 5120 --sweep 30:56 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 248320 --k 5120 --sweep 30:56 --repeat 50
./build/bench/ninfer_linear_pair_bench --tokens 64,128,192,193,256,384,512,768,1024 --repeat 50
./build/bench/ninfer_gdn_input_proj_bench --repeat 50
./build/bench/ninfer_attn_input_proj_bench --repeat 50
./build/bench/ninfer_sparse_moe_bench --codec all --tokens 1024|4096|8192 --cache cold --execution eager --repeat 50
```

The MoE benchmark takes `--execution eager` because prefill is not graph captured: the only capture
site is `src/core/decode_graph.cpp` and the prefill chunk loop is outside it.

### The changed kernel, and the benchmark's own floor

`w8_linear_add --production-only` covers four kernels and only one is this one, so rows are grouped
by the kernel each shape reaches:

| kernel | rows | T | median | best | worst |
|---|---:|---|---:|---:|---:|
| `w8_rowsplit_gemm_mma` | 18 | 129-1024 | **x1.123** | x1.211 | x1.018 |
| `w8_small_t_mma` (`splitk8`) | 21 | 2-48 | x1.000 | x1.001 | x0.999 |
| `medium_splitk` | 10 | 63-128 | x1.000 | x1.043 | x0.960 |
| decode | 1 | 1 | x0.857 | - | - |

Three of those four kernels are not compiled differently by this branch, so their rows are the
harness floor. Rather than assert that the floor is wide, I measured it: three independent passes
of the same benchmark against the same binary, per arm.

| benchmark | arm | rows | median pass-to-pass spread | p90 | worst |
|---|---|---:|---:|---:|---:|
| `w8_linear_add --production-only` | master | 50 | x1.000 | x1.035 | x1.161 |
| `w8_linear_add --production-only` | branch | 50 | x1.000 | x1.026 | x1.077 |
| `w8_linear_swiglu --production-only` | master | 39 | x1.000 | x1.031 | x1.100 |
| `w8_linear_swiglu --production-only` | branch | 39 | x1.000 | x1.043 | x1.100 |

The worst row of the master arm is `linear_add.w8.decode.r16.residual` at T=1 - the same row that
reads x0.857 in the table above - at x1.161 against itself. Every figure below unity anywhere in
this PR sits on a row whose own repeat spread is at least as wide, and every one of them is on a
kernel this branch does not change.

### The other five Op families

| benchmark | rows that move by >1% | median of those | best | worst |
|---|---:|---:|---:|---:|
| `w8_linear_swiglu --production-only` | 20 of 39 | **x1.099** | x1.143 | x0.909 |
| `linear --suite all` | 20 of 68 | **x1.089** | x1.171 | x0.965 |

The `linear` suite carries Q4, Q5, Q6, BF16, NVFP4, FP8 and W8 rows. Every mover except one is a W8
row; the exception is a Q5 `vision_fc2` row at T=1024 at x0.985. One W8 row moves down,
`27b.mtp_attention` at T=1 (x0.965), and 14 of the 33 W8 rows sit within one percent of unity. The
two `linear_swiglu` rows below unity are `splitk.mma.pair.exact_t` at T=3 and T=11 - the two rows
the repeat table puts at x1.100 and x1.091 against themselves.

`linear_pair` puts a route boundary on display: `kK2048Routes` first selects a `ConcatMma` schedule
that instantiates this header at T=193.

| T | 64 | 128 | 192 | 193 | 256 | 384 | 512 | 768 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| speedup | x1.000 | x1.000 | x1.007 | **x1.136** | **x1.133** | **x1.200** | **x1.118** | x1.042 | x1.068 |

The two projection benchmarks measure one operator in several weight formats, so the change and the
floor sit side by side:

| operator | format | T=64 | 128 | 256 | 512 | 1024 |
|---|---|---:|---:|---:|---:|---:|
| `gdn_input_proj` | **w8** | x1.000 | **x1.065** | **x1.109** | **x1.097** | **x1.073** |
| `attn_input_proj` | **w8-qkv** | x1.000 | **x1.105** | **x1.159** | **x1.102** | **x1.099** |
| `attn_input_proj` | **w8-qgkv** | x0.991 | **x1.044** | **x1.067** | **x1.053** | **x1.093** |
| both | bf16, fp8, nvfp4, q4q5 | x1.000 | x1.000 | x1.000-1.031 | x0.993-1.016 | x0.989-1.000 |

### The GROUPS == 4 schedule

Swept at `k=5120` across both of its route boundaries. The speedup appears exactly where the
dispatch table hands the shape to this kernel and disappears below it, which is the check that the
sweep measures what it claims to:

| n | T=30-33 | T=34-40 | T=41-48 | T=49-56 |
|---|---|---|---|---|
| 34 816 | x1.000-1.013 (`small_t`) | x1.000-1.012 (`small_t`) | **x1.085-1.100** (`r64x16_c48_k128_a1`) | x1.058 (`mma_r64_c128`) |
| 248 320 | x0.992-0.998 (`small_t`) | **x1.146-1.187** (`r64x16_c48_k128_a1`) | **x1.081-1.185** (same) | x1.066-1.073 (`mma_r64_c128`) |

The two rows part company exactly where the dispatch table says they should: `small_t` runs to T=40
at `n=34816` and only to T=33 at `n=248320`. Both `T=49-56` columns are this kernel again on its
wide schedule.

### Confinement, measured rather than argued

The routed MoE prefill benchmark - the kernels of the sibling change, which this branch does not
touch:

| codec | 1024 | 4096 | 8192 |
|---|---:|---:|---:|
| q4-q5 | x0.997 | x0.999 | x1.000 |
| q4-q6 | x0.997 | x1.000 | x0.999 |
| w8-w8 | x1.000 | x0.999 | x0.998 |

And the same question at product scale: `nsys` kernel totals for one 33,031-token prefill at chunk
8192, both arms, same command.

| kernel group | master | share | branch | speedup |
|---|---:|---:|---:|---:|
| `w8_rowsplit_gemm_mma` (3 forms) | 488.8 ms | 25.9% | 452.4 ms | **x1.0805** |
| `causal_attention_prompt_bf16` | 493.7 ms | 26.1% | 493.8 ms | x0.9999 |
| `sparse_moe_prefill_q4_gate_up` | 360.1 ms | 19.0% | 360.0 ms | x1.0001 |
| `sparse_moe_prefill_qx_down` | 194.8 ms | 10.3% | 194.8 ms | x1.0001 |
| gated delta net (4 kernels) | 109.8 ms | 5.8% | 109.8 ms | x1.0003 |
| `sparse_moe_prefill_gather` | 28.8 ms | 1.5% | 28.8 ms | x0.9997 |
| **all kernels** | **1890.8 ms** | | **1854.5 ms** | **x1.0196** |

The in-product speedup on the changed kernel is x1.081 rather than the x1.12 the cold-L2 benchmark
reports, which is what a warm L2 and a mix containing the narrow schedules should do. Every other
group sits inside 0.03% of unity. The total, x1.0196, is the +1.97% the end-to-end table below
reports at the same point.

### Resources

`cuobjdump --dump-resource-usage`, this branch against `master`:

| kernel | instantiations | registers max | registers mean | shared | spills |
|---|---:|---:|---:|---|---|
| `w8_rowsplit_gemm_mma` | 120 | 127 -> **117** | 102.4 -> **96.2** | identical set of 22 values | none either side |
| MoE prefill gate/up and down | 19 | 116 -> 116 | 70.6 -> 70.6 | unchanged | none either side |

Registers fall in 103 of the 120 and rise in 9. Workspace, resident memory, transfers and graph
nodes are unchanged: no allocation is added or removed and no boundary moves. The MoE row is there
because those kernels are #106's subject; this branch leaves them untouched, register for register.

`cuobjdump -sass`, compared body by body. Each binary carries 2931 device functions; 2893 match by
name, and the 38 that do not are the internal-linkage instantiations from
`src/ops/linear_pair/w8/w8_pair_gemm_concat.cu`, whose symbols carry a per-compilation module id -
exactly the instantiations behind the `linear_pair` row above, which is where their behaviour is
measured instead. Of the 2893 matched, **120 differ and 2773 are identical in both instruction
count and per-opcode census**, and all 120 are instantiations of this kernel.

| instantiation | instructions | STS | LDS | I2F | HMMA |
|---|---:|---:|---:|---:|---:|
| deepest reduction | 1976 -> 1336 (-32.4%) | 72 -> 37 | 61 -> 11 | 80 -> 10 | 160 -> 160 |
| most instructions | 2616 -> 1984 (-24.2%) | 40 -> 5 | 60 -> 10 | 80 -> 10 | 160 -> 160 |
| BN=128 form | 1920 -> 1448 (-24.6%) | 30 -> 5 | 45 -> 10 | 60 -> 10 | 160 -> 160 |

Counts fall between 1.2% and 32.4%, median 4.9%; most instantiations are narrow schedules where the
staging loop is a small share of the body. Summed across all 120: `SHFL` **1150 -> 0**, `STS`
2884 -> 1724, `LDS` 2934 -> 1514, `I2F` 3560 -> 1240, `LOP3` 18003 -> 12982, `IMAD` 16476 -> 15493,
`HMMA` 15120 -> 15120 with **zero bodies in which the HMMA count moves**. One counted opcode rises,
`F2FP` 3994 -> 4694. These are static counts in differently unrolled bodies rather than per-weight
rates, so the one that carries the claim is `HMMA`: the tensor-core arithmetic is untouched and what
gets cheaper is the preparation of its operands.

### Numerics

Per weight the arithmetic is the same int8 sign extension, the same multiply by the same group
scale and the same `__floats2bfloat162_rn`. Only the route by which the scale reaches the thread
changes. For `GROUPS == 2` the old code loaded a `uint32` at `scale_pair_offset` and extracted
`>> ((gg & 1) * 16)`; for `GROUPS == 4` it shuffled two `uint32` from `+0` and `+4`. Both resolve to
the halfword at `scale_pair_offset + gg * 2`, which is what the thread now reads directly. The
highest byte touched is 16, which is `SCALE_CACHE_BYTES`.

Removing the shuffles is safe because they sat in a warp-uniform loop: the old trip count depended
only on `warp`, so all 32 lanes were converged.

The shipped Op tests compare against tolerances rather than bits, so what establishes bit-identity
is the `HMMA` census above together with greedy output byte-identical to `master` at every measured
point. The `GROUPS == 4` schedule is reached by `tests/ops/linear/test_w8_a16.cpp` at t=34, t=41 and
t=48 through `src/ops/linear/w8/w8_dispatch.cpp:31-38`.

```bash
cd build && ctest -j1
```

92 pass, 1 skipped, 1 fails - on this branch and on `master` built in the same directory in the same
campaign. The skip is `27b_load_plan`, which requires both real 27B artifacts and only one is on
this box. The failure is `ninfer_qwen3_6_27b_prefix_real_test`, with the same message on both arms:
`Host checkpoint restore changed greedy output: restored=64,1248, baseline=64,56127 ...
transfers=1/1/3/3`. It is #105 and still open.

### End to end

Confirmation only. Arms alternate inside each round, every point its own process, greedy, four
rounds. The spread column is the largest round-to-round spread among the arms in that row.

| model | chunk | tokens | n | prefill | decode | spread |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 8192 | 8 515 | 4 | **+2.34%** | +0.03% | 0.49% |
| Qwen3.6-35B-A3B | 8192 | 33 031 | 4 | **+1.97%** | -0.01% | 0.50% |
| Qwen3.6-35B-A3B | 8192 | 65 882 | 4 | **+1.56%** | +0.01% | 0.29% |
| Qwen3.6-35B-A3B | 1024 | 33 031 | 4 | **+1.73%** | +0.01% | 0.16% |
| Qwen3.6-27B dense | 8192 | 33 031 | 4 | -0.05% | -0.01% | 0.44% |

At every 35B point the slowest of the four candidate runs is faster than the fastest of the four
baseline runs, so the separation does not depend on round ordering. All 20 generations are
byte-identical to `master`. The gain falls with prompt length because attention's share of prefill
grows while the GEMM's falls, and the change lives only in the GEMM.

The dense 27B row is close to a null result rather than a control: its W8 row-split tensors are the
MTP and vision entries, and a text-only greedy run with speculation off does not reach them.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host, so occupancy and memory-pipe
  counters could not be collected. The register and shared-memory figures come from
  `cuobjdump --dump-resource-usage` and the instruction counts from `cuobjdump -sass` instead.
- `docs/performance.md` publishes prefill throughput for `qwen3_6_35b_a3b` measured at a 1,024-token
  prefill chunk with INT8 group-64 KV, a path this change is on, so those figures would move. I have
  not re-run that harness - it uses prefix reuse, CUDA Graphs and five seeds through the serve corpus
  - and I am not proposing edits to the document.
- Speculation is off in every run here, so the MTP leaves are not exercised.
- Qwen3.8-27B NVFP4 is not measured; it takes no W8 row-split route.
- The 38 concat instantiations are covered by the `linear_pair` measurement and the greedy-output
  gate, not by the instruction census.
