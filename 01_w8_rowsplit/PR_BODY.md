Implements #ISSUE.

Scope as confirmed in #ISSUE: …

`w8_rowsplit_gemm_mma`'s `dequant_w` staged the weight tile one 16-bit code pair per lane: each lane
decoded two codes and issued a 4-byte `store_vec`, so a row of the tile cost 32 four-byte shared
stores on the `BK == 64` schedules and 64 on the `BK == 128` one. A lane also could not read its own
quantisation-group scale, because its pair might belong to a group whose scale another lane had
loaded, so the scales were broadcast with `__shfl_sync` - once per row pair on the `GROUPS == 2`
branch and twice on `GROUPS == 4`.

Each thread now owns a whole eight-code run inside one group: it reads its own scale halfword,
decodes eight codes out of one `uint2` and stores them with a single 16-byte `store_vec`. Per weight
the arithmetic is unchanged, so this changes how operands are prepared, not what is computed.

This is the projection half of the widening whose MoE half you merged as #106. It is based on
`master` at `1fc1cb76`, which already carries that half; `dev` is at `1fc1cb76` too, so nothing
unmerged touches this header.

## The design

The loop is indexed by thread over eight-code chunks rather than by row pair over lanes.

- **A thread owns a chunk, so it owns the scale.** The three `shfl_sync` call sites go, and with them
  the reason the loop had to be built around `lane >> 4` and `lane & 15`. The scale loads change
  shape rather than growing: every thread now loads instead of two or four lanes per warp, but the
  number of load instructions per tile halves on `BK == 64` and is unchanged on `BK == 128`, and each
  touches four distinct 32-bit words in four distinct banks, so it stays conflict-free. On a single warp a deliberate four-way conflict costs 12.64 cycles
  per load and a true broadcast 8.39; this access pattern sits at 8.76 and 8.78, within 5% of
  broadcast, which is address arithmetic rather than conflict.
- **Eight decoded codes are sixteen contiguous bytes**, so four 4-byte stores become one 16-byte
  store, through `W8Bf16x8Bits` - the union this file already keeps for that shape and already uses
  in the residual epilogue, in the same write-`pair`, read-`raw` direction.
  `w8g32_swz64(row, col) = (((col >> 3) ^ (row & 7)) << 3) | (col & 7)`, so at `col = 8c` the low
  three bits vanish, the offset is a multiple of eight elements and therefore 16-byte aligned, and
  `c ^ (row & 7)` is a bijection on a row's chunks: the eight elements land in the same eight slots
  in the same order as before.
- **The relations the loop shape rests on are stated.** Eight is the widest chunk contiguous in `As`
  *for every row* - a wider run is contiguous only when `row & 7` is even. Beyond that the body needs
  `BK % 32 == 0`, so `gg = col >> 5` stays inside `GROUPS` and the row stride stays aligned for the
  vector store, and `8 % GROUPS == 0`, so the scale cache holds whole tiles. The `GROUPS == 4`
  assertion the old code carried on its `else` path goes with that branch, so those two relations are
  asserted where the loop now lives.

## Affected behaviour and ownership

One function in `src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh`, +25/-35. No public header, no CLI or
serving surface, no artifact contract, no workspace or arena boundary, no graph node, no new
environment variable, no `getenv`, no `thread_local`, no module-global device state. Numerically the
change is bit-for-bit identical, established by measurement below.

The header is instantiated by six public Op families - Linear, LinearAdd, LinearPair, LinearSwiGLU,
AttnInputProj and GdnInputProj - so all six are measured. It is not a prefill-only change: the only
`GROUPS == 4` schedule, `MmaR64x16C48K128A1`, serves `k=5120, n=34816` at `t` in [41,48] and
`k=5120, n=248320` at `t` in [34,48] (`src/ops/linear/w8/w8_dispatch.cpp:30-39`), and the projections
enter this kernel at T=65 and T=97. Those are decode-shaped extents, reachable as a speculative batch
under the one-to-eight active requests the product targets.

## Verification

The claim is at operator level. The end-to-end section is confirmation, not the evidence.

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, gcc 13.3, Ubuntu 24.04, Release,
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Base `1fc1cb76`. The figures were taken on `3a61ef3f`; the four commits since then touch `src/serve`,
the cache test oracle and the FP8 vocabulary GEMM, none of them this header or its consumers, and
the suite and the instruction census were re-run on the new base. Both arms built in the same build
directory, the same two binaries throughout. Each benchmark's own default warmup (3 for `linear`, 5 for the others)
and 50 measured samples. L2 is flushed for every sample; the timing path is
`bench::measure_cold_launch` except for `linear_pair`, which is cold **graph replay** through
`bench::measure_cold_graph`, and `sparse_moe`, which its own harness times eagerly - prefill is not
graph captured, the only capture site being `src/core/decode_graph.cpp` with the chunk loop outside
it.

Everything below was run at least twice: two independent campaigns with separate builds of both arms,
three passes for the production-extent table and for the repeat-floor and projection tables, and five
for `linear_pair`. The one exception is the 512-token output gate, a single run per arm. Tables quote the second campaign; where the
campaigns differ it is stated. All ratios are master / branch, so above 1.000 is faster.

```bash
./build/bench/ninfer_w8_linear_add_bench --production-only --repeat 50
./build/bench/ninfer_w8_linear_swiglu_bench --production-only --repeat 50
./build/bench/ninfer_linear_bench --suite all --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 34816  --k 5120 --sweep 30:56 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 248320 --k 5120 --sweep 30:56 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 12288 --k 2048  --t 1024|2048|4096|8192 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 9216  --k 2048  --t 1024|2048|4096|8192 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 2048  --k 4096  --t 1024|2048|4096|8192 --repeat 50
./build/bench/ninfer_linear_bench --qtype W8 --n 2048  --k 16384 --t 1024|2048|4096|8192 --repeat 50
./build/bench/ninfer_linear_pair_bench --tokens 64,128,192,193,256,384,512,768,1024 --repeat 50
./build/bench/ninfer_gdn_input_proj_bench  --tokens 32,64,65,96,97,128,256,512,1024 --repeat 50
./build/bench/ninfer_attn_input_proj_bench --tokens 32,64,65,96,97,128,256,512,1024 --repeat 50
./build/bench/ninfer_sparse_moe_bench --codec all --tokens 1024|4096|8192 --cache cold --execution eager --repeat 50

cuobjdump --dump-resource-usage ./build/apps/ninfer
cuobjdump -sass ./build/apps/ninfer

nsys profile -t cuda --cuda-graph-trace=node -o p --force-overwrite=true \
  ./build/apps/ninfer qwen3_6_35b_a3b.ninfer --messages <32k-prompt.json> --max-new 1 --greedy \
  --no-thinking --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o k p.nsys-rep

./build/apps/ninfer qwen3_6_35b_a3b.ninfer --messages <prompt.json> --max-new 32|512 --greedy \
  --no-thinking --max-context 131072 --prefill-chunk 8192|1024 --kv-dtype bf16
```

Lines carrying `|` are that many separate runs. Artifacts are `qwen3.6-35b-a3b/groupwise-int` and
`qwen3.6-27b/groupwise-int`; the prompts are three needle-in-a-haystack documents of 8,515 / 33,031 /
65,882 tokens under the registered chat template.

### The operator, at the extents the product runs

The shipped benchmark suites stop at T=1024, and prefill runs at chunk 8192, so the headline figure
further down is a median over shapes and extents the model does not execute. Measured directly at
the four W8 shapes a 35B prefill actually uses, at the extents it uses them:

| n, k | T=1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| 12288, 2048 | 1.080-1.086 | 1.086-1.090 | 1.087 | 1.087-1.089 |
| 9216, 2048 | 1.093-1.103 | 1.090 | 1.088-1.089 | 1.088-1.090 |
| 2048, 4096 | 1.071-1.091 | 1.083 | 1.096-1.100 | 1.089-1.092 |
| 2048, 16384 | 1.089 | 1.118-1.121 | 1.096-1.099 | 1.093-1.095 |

Sixteen shapes and extents, three independent passes each, so each cell is the range over its three
passes. **Median x1.090 over the 48 points, range x1.071 to x1.121.** That is the operator figure the
end-to-end result rests on, and it matches the in-product kernel measurement below (x1.082) rather
than the x1.12 the `linear_add` production sweep reports at T=129-1024.

### The changed kernel in the shipped suites, against the harness floor

`w8_linear_add --production-only` covers four kernels and only one is this one, so rows are grouped
by the kernel each shape reaches:

| kernel | rows | T | median | best | worst |
|---|---:|---|---:|---:|---:|
| `w8_rowsplit_gemm_mma` | 18 | 129-1024 | **x1.123** | x1.211 | x1.018 |
| `w8_small_t_mma` (`splitk8`) | 21 | 2-48 | x1.000 | x1.000 | x0.938 |
| `medium_splitk` | 10 | 63-128 | x1.000 | x1.026 | x0.960 |
| decode | 1 | 1 | x0.859 [^1] | - | - |

[^1]: a single T=1 cell on a kernel this branch does not compile differently; the same row varies by
x1.164 against itself on one arm, see the repeat-pass table below.

Three of those four are not compiled differently by this branch, so their rows are the harness floor
rather than a control for the change, and it is not a tight floor. Three things measure it rather
than asserting it.

**Three passes of the same benchmark against the same binary**, per arm, as max/min per row, p90 by
nearest rank:

| benchmark | arm | rows | median | p90 | worst |
|---|---|---:|---:|---:|---:|
| `w8_linear_add` | master | 50 | x1.000 | x1.002 | x1.098 |
| `w8_linear_add` | branch | 50 | x1.000 | x1.025 | x1.164 |
| `w8_linear_swiglu` | master | 39 | x1.000 | x1.076 | x1.100 |
| `w8_linear_swiglu` | branch | 39 | x1.000 | x1.044 | x1.100 |

The worst row is `linear_add.w8.decode.r16.residual` at T=1, x1.164 against itself on the branch arm
- the same row that reads x0.859 across arms.

**The two campaigns disagree about which rows dip.** Campaign 1 put the sub-unity rows on
`27b.mtp_attention` T=1, a Q5 `vision_fc2` row, and two `linear_swiglu` `exact_t` rows at T=3 and T=11
(x0.909 and x0.918); campaign 2 puts them on `27b.mtp_down` T=16 (x0.972), `35b.dflash_feature` T=16
(x0.957) and a Q6 `vision_patch` row at T=128 (x0.951), with no `linear_swiglu` row below unity at
all. The intersection is empty, and in neither campaign does any of them land on the changed kernel.

**And a large share of the flat cells are flat to the harness's resolution.** Every median the
benchmarks report is a multiple of 32 ns, and the two arms produce a byte-identical median in 25 of
50 `w8_linear_add` rows, 26 of 48 `gdn_input_proj` cells and 51 of 72 `attn_input_proj` cells. Where
this PR prints x1.000 it usually means the two medians were the same number, not that a difference
was resolved and found small.

### The other five Op families

`linear --suite all` carries Q4, Q5, Q6 and W8 rows only - the BF16, NVFP4 and FP8 kernels are
reachable in that benchmark only behind an explicit `--qtype` and were not run there; they are
covered in the projection benchmarks below instead. Of its 68 rows, 21 move by more than one percent
either way, median x1.089, best x1.171, worst x0.951, and 20 of the 21 are W8 rows.

Splitting the 33 W8 rows by extent says more than that aggregate does:

| W8 rows | count | median | best | worst |
|---|---:|---:|---:|---:|
| T = 128 and T = 1024, the extents that reach this kernel | 17 | **x1.089** | x1.171 | x1.059 |
| T = 1, 4, 16, decode extents on other kernels | 16 | x1.000 | x1.066 | x0.957 |

**All 17 rows at the extents that reach this kernel move up**, none by less than 5.9%. The 16 short
rows are the floor, and one shape shows on its own what that floor is worth:
`35b.dflash_feature` (n=2048, k=16384) reads **x1.066 at T=1 and x0.957 at T=16** - ten points apart,
across two decode extents of a route this branch does not compile differently. The only other row
below unity is `27b.mtp_down` at T=16, x0.972.

| benchmark | rows moving >1% either way | median of those | best | worst |
|---|---:|---:|---:|---:|
| `w8_linear_swiglu --production-only` | 18 of 39 | **x1.105** | x1.143 | x1.038 |

`linear_pair`, where `kK2048Routes` (`src/ops/linear_pair/w8/w8_pair_plan.cpp:30-43`) first selects a
`ConcatMma` schedule that instantiates this header at T=193. Five independent passes; the range is
what five passes gave, not an error bar:

| T | 64 | 128 | 192 | 193 | 256 | 384 | 512 | 768 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| range over 5 passes | 0.984-1.016 | 1.000-1.011 | 0.994-1.061 | **1.136-1.200** | **1.133-1.143** | **1.200** | **1.118** | 1.042-1.043 | 1.068-1.069 |

The on-route cells from T=193 up are stable to a thousandth across passes; the off-route cells below
it are not, and T=192 spans 0.994 to 1.061 on a kernel this branch does not compile differently. The
T=193 spread is on the master arm, whose median there is bimodal - 34.85 us in one pass, 36.64 in another and 36.83 in the
remaining three.

### Where the routes hand the shape over

Three dispatch tables put a boundary inside a sweep, and the speedup starts at each one.

`k=5120` W8 linear, `--sweep 30:56`:

| n | T=30-33 | T=34-40 | T=41-48 | T=49-56 |
|---|---|---|---|---|
| 34 816 | x1.000 `small_t` | x0.989-1.000 `small_t` | **x1.075-1.100** `r64x16_c48_k128_a1` | **x1.057-1.065** `mma_r64_c128` |
| 248 320 | x1.000-1.004 `small_t` | **x1.149-1.187** `r64x16_c48_k128_a1` | **x1.085-1.184** same | **x1.074-1.084** `mma_r32_c64` |

`small_t` runs to T=40 at `n=34816` and only to T=33 at `n=248320`, exactly as `w8_dispatch.cpp:30-39`
says, and the ratio steps at those two different extents accordingly. Above T=48 the two shapes take
different wide schedules of this same kernel, so three of its schedules are visible in one sweep, all
moving, with the untouched route flat underneath. One thing worth flagging that the ratios hide: on
`master` the `GROUPS == 4` route is *slower* at its own boundary than the `small_t` route it replaces
at its first extent (T=41, 206.6 us) is slower than the `small_t` route it takes over from at the
last extent that route serves (T=40, 194.6 us); this change removes that inversion (T=41, 190.5 us).
Those three are campaign 1; campaign 2 reads 206.8, 192.5 and 190.5.

The projections, two passes, both shown. `gdn_input_proj` hands over at T=97
(`src/ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp:20-23`); `attn_input_proj` at T=65 for the target
and T=97 for the companion (`src/ops/attn_input_proj/w8/w8_attn_input_plan.cpp:20-25`, `:27-37`).
Below those extents all three w8 rows are on `SplitKMmaDirect`, untouched here:

| operator | 32 | 64 | 65 | 96 | 97 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gdn_input_proj` w8 | 1.000 | 1.000 | 1.000 | 1.000 | **1.062 / 1.031** | 1.065 | 1.085 / 1.097 | 1.097 | 1.066 |
| `attn_input_proj` w8-qkv | 1.000 | 1.000 | 1.000 / 1.067 | 1.000 | **1.105 / 1.105** | 1.105 | 1.160 | 1.075 / 1.103 | 1.099 |
| `attn_input_proj` w8-qgkv | 1.000 | 0.999 / 1.053 | **1.045 / 1.045** | 1.049 / 1.091 | 1.043 / 1.092 | 1.044 | 1.067 | 1.071 | 1.093 |

Two cells below the handover move between passes - `w8-qkv` at T=65 and `w8-qgkv` at T=64. Both are
off-route and inside the short-extent floor. Every on-route cell moves up in both passes; three of
them differ between passes - `gdn_input_proj` at T=97 (1.062 / 1.031) and T=256 (1.085 / 1.097), and
`w8-qgkv` at T=97 (1.043 / 1.092) - which is the same short-extent variance the floor rows show.

The seven untouched weight-format rows in these two benchmarks - `q4q5`, `nvfp4`, `fp8` on
`gdn_input_proj` and those plus `bf16` on `attn_input_proj` - span x0.958 to x1.053 below T=64 and
x0.969 to x1.031 at T=64 and above, with the widest cells being nvfp4 `gdn_input_proj` at T=256 and
T=1024. That is the floor these benchmarks carry, and the w8 rows have to be read against it.

Warm L2 was not the explanation for anything here: run with `--cache both`, the warm ratios are
indistinguishable from cold (`gdnip w8` T=256 cold x1.109 / warm x1.114; `attnip w8-qkv` T=256 cold
x1.159 / warm x1.169).

### Confinement

The routed MoE prefill benchmark - #106's kernels, untouched here:

| codec | T=1024 | T=4096 | T=8192 |
|---|---:|---:|---:|
| q4-q5 | x0.998 | x0.999 | x1.000 |
| q4-q6 | x0.997 | x1.000 | x0.998 |
| w8-w8 | x1.000 | x0.999 | x0.999 |

And at product scale. `nsys` kernel totals for one 33,031-token prefill at chunk 8192:

| kernel group | master | branch | speedup | share of master |
|---|---:|---:|---:|---:|
| `w8_rowsplit_gemm_mma`, 6 instantiations | 491.85 ms | 454.75 ms | **x1.082** | 25.84% |
| `causal_attention_prompt_bf16` | 495.63 | 495.43 | x1.000 | 26.03% |
| `sparse_moe_prefill_q4_gate_up` | 364.30 | 363.90 | x1.001 | 19.13% |
| `sparse_moe_prefill_qx_down` | 197.42 | 197.08 | x1.002 | 10.37% |
| gated delta net, 7 kernels | 110.24 | 110.10 | x1.001 | 5.79% |
| `sparse_moe_prefill_gather` | 28.94 | 28.94 | x1.000 | 1.52% |
| the remaining 46 kernels | 215.62 | 215.80 | x0.999 | 11.32% |
| **all 67 kernels** | **1904.00** | **1866.00** | **x1.020** | 100% |

Four profile pairs were taken across the two campaigns. Across all four, no unchanged group deviates
from unity by more than **0.27%** (worst: `qx_down` 0.273%, `q4_gate_up` 0.249%, attention 0.133%,
gdn 0.125%, gather 0.071%). A single pair looks tighter than that, but repeating it does not support
a tighter bound. Checked individually rather than trusting the grouping, the widest move among single
unchanged kernels longer than 1 ms is x0.990 in the pair above and x1.013 across all four - both
times `sigmoid_gate_mul_bf16x8_kernel`, 3.4 ms.

The launch count of every one of the 67 kernels is identical between arms, which is the direct
evidence for "no graph node appears or disappears".

The in-product speedup on the changed kernel is x1.082 in the pair above and x1.079 to x1.082 across
all four pairs, against the x1.089 the operator measurement at
production extents gives - not against the x1.12 the `linear_add` sweep reports at extents the model
does not run. The total, x1.020 in this pair and x1.018 to x1.021 across the four, is not independent
evidence: GPU busy time is 99.5% of prefill wall
clock here, so the kernel sum and the end-to-end prefill figure are the same quantity read off two
clocks.

### Resources

The binary carries 158 instantiations of this kernel. 120 match by name between the two builds; the
other 38 are the internal-linkage instantiations from `src/ops/linear_pair/w8/w8_pair_gemm_concat.cu`,
whose symbols carry a per-compilation module id - exactly the instantiations behind the `linear_pair`
row above, which is where they are measured instead.

| kernel | matched | registers max | registers mean | shared | spills |
|---|---:|---:|---:|---|---|
| `w8_rowsplit_gemm_mma` | 120 of 158 | 127 -> **117** | 102.4 -> **96.2** | same set of 22 values | none either side |
| MoE prefill gate/up and down | 19 | 116 -> 116 | 70.6 -> 70.6 | unchanged | none either side |

Registers fall in 103, rise in 9, unchanged in 8. For reference against #106's table, which reported this kernel on master as 158 instantiations with
maximum 128: `cuobjdump --dump-resource-usage` over the master binary splits those 158 into the 120
named instantiations, whose maximum is 127, and the 38 internal-linkage concat symbols, whose
maximum is 128. So the single 128-register instantiation is one of the 38, which is why the
matched-set maximum reads 127 here. The MoE row is there
because those kernels are #106's subject; this branch leaves them untouched, register for register.
Workspace, resident memory, transfers and graph nodes are unchanged.

`cuobjdump -sass`, compared body by body. Of the 2893 name-matched device functions, **120 differ in
instruction count or per-opcode census and 2773 are identical in both**, and all 120 are
instantiations of this kernel. Over those 120 bodies the instruction count falls 179,948 -> 161,238
(-10.4%), per body between 1.6% and 33.3%, median 4.9%.

The whole shuffle machinery leaves: `SHFL` **1150 -> 0**, `WARPSYNC` **1360 -> 0**,
`ENDCOLLECTIVE` **390 -> 0**. Staging and the integer work around it fall: `STS` 2884 -> 1724,
`LDS` 2934 -> 1514, `MOV` 12215 -> 8466, `UMOV` 4444 -> 2481, `S2R` 849 -> 335, `S2UR` 1351 -> 531,
`LOP3` 17924 -> 12940, `IADD` 11998 -> 9701, `IMAD` 16476 -> 15493. Some rise, and the two that read against the
narrative are named first: `BSSY` and `BSYNC` both 2667 -> 2967, the reconvergence barriers, because
one wide loop replaces the row-pair loop and its nesting. Then `SHF` 5561 -> 7649, `FMUL`
3976 -> 5376, `F2FP` 3994 -> 4694, and the int-to-float conversion moves to a packed form, `I2F`
3560 -> 1240 with `I2FP` 0 -> 3720.

What does *not* move is the argument. These opcodes are identical, body for body, across all 120:
**`HMMA` 15120**, `LDSM` 12480, `LDGSTS` 1802, `FFMA` 4160, `STG` 1882, `LDG` 210, `BAR` 1302,
`DEPBAR` 600, `MUFU` 832. The tensor-core stream, the shared-memory matrix loads that feed it, the
global staging, the epilogue stores and the barriers are untouched in every instantiation; what gets
cheaper is the preparation of the operands. There is no body in which the `HMMA` count moves.

| instantiation | instructions | STS | LDS | I2F | I2FP | SHFL | HMMA |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepest reduction, `BM=64 BN=48 BK=64` | 1922 -> 1282 (-33.3%) | 72 -> 37 | 61 -> 11 | 80 -> 10 | 0 -> 30 | 30 -> 0 | 160 -> 160 |
| largest body, `BM=128 BN=64 BK=64` | 2913 -> 2466 (-15.3%) | 46 -> 21 | 61 -> 26 | 60 -> 10 | 0 -> 30 | 20 -> 0 | 160 -> 160 |

The census parser matters here: the one shipped with our earlier campaign scripts folded `PLOP3`
into `LOP3` and `I2FP` into `I2F`, because its instruction regex mis-handles a uniform predicate.
The numbers above come from a parser that strips `@!?U?P\d+` explicitly.

### Numerics

Per weight the arithmetic is the same int8 sign extension, the same multiply by the same group scale
and the same `__floats2bfloat162_rn`. Only the route by which the scale reaches the thread changes:
for `GROUPS == 2` the old code read a `uint32` at the scale tile offset and extracted
`>> ((gg & 1) * 16)`, and for `GROUPS == 4` it shuffled two `uint32` from `+0` and `+4`. Both resolve
to the halfword at `scale_tile_offset + gg * 2`, which is what the thread now reads directly, and the
highest byte index touched is 15, the last of the 16 in `SCALE_CACHE_BYTES`, as before. Removing the
shuffles is safe because they sat in a warp-uniform loop: the old trip count depended only on `warp`,
so all 32 lanes were converged, confirmed by reading `activemask` at every scale-load site in both
bodies.

What establishes bit-identity is a decision benchmark rather than that argument. Two harnesses, both
throwaway and not part of this PR:

1. The old and new `dequant_w` bodies side by side in one CTA, over all 26 distinct
   `(BM, BK, THREADS)` instantiations that reach this kernel, every `kt % SCALE_CACHE_TILES` residue,
   random codes and adversarial FP16 scale bit patterns - signed zeros, subnormals, both infinities,
   quiet and signalling NaN, max normal - with a sentinel in every shared slot to catch one the loop
   fails to write. **Zero mismatches, zero unwritten slots.**
2. Whole-kernel A/B from two binaries, output tensors byte-compared over 33 cases: `Full` and masked,
   the Store, Residual and SwiGLU epilogues, both SwiGLU branches, `BK` 64 and 128, ragged `m`, `n`
   and `k`, `k < padded_k`, and a `padded_k` that forces a mid-loop scale-cache refill.
   **983,163 bytes identical.**

`compute-sanitizer` memcheck, racecheck, synccheck and initcheck report the same counts on both arms.

The shipped Op tests qualify the routes against an independent FP64 oracle (`cpu_linear_gemm_fp64`)
with `Comparison::Sampled`, so they check a sample of outputs against a tolerance rather than every
output bit. Their shape lists do step across every boundary this change touches: `kN248320K5120`
carries `a16(33), a16(34), a16(48), a16(49), a16(64), a16(65)` and `kN34816K5120` carries
`a16(40), a16(41), a16(48), a16(49)`.

```bash
cd build && ctest -j1
```

**94 tests, 0 failed, 1 skipped**, on master and on this branch, on the rebased base. The skip is
`27b_load_plan`, which needs both real 27B artifacts and only one is on this box.

On `3a61ef3f` this suite had one failure, `ninfer_qwen3_6_27b_prefix_real_test`, which we reported
as #105. `fd48e2fa` between the two bases rewrites that test's oracle, so it now passes; the
submission no longer carries a red test.

### End to end

Confirmation only. Arms alternate inside each round, one process per point, greedy, four rounds.
Prefill and decode are percent faster than master, higher is better.

| model | chunk | prompt tokens | rounds | prefill | decode | round spread |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 8192 | 8 515 | 4 | **+2.41%** | -0.01% | 0.34% |
| Qwen3.6-35B-A3B | 8192 | 33 031 | 4 | **+1.98%** | +0.01% | 0.28% |
| Qwen3.6-35B-A3B | 8192 | 65 882 | 4 | **+1.59%** | -0.00% | 0.17% |
| Qwen3.6-35B-A3B | **1024** | 33 031 | 4 | **+1.74%** | +0.01% | 0.11% |
| Qwen3.6-27B dense | 8192 | 33 031 | 4 | -0.04% | -0.05% | 0.42% |

1,024 is the product default and its row is the +1.74% line; the 8192 rows are the additional point,
where the `nsys` attribution was also taken. At every 35B point the slowest of the four candidate runs
is faster than the fastest of the four baseline runs, in both campaigns, so the separation does not
depend on round ordering. The gain falls with prompt length because attention's share of prefill
grows while the GEMM's falls, and the change lives only in the GEMM. The dense 27B row is a null
result rather than a control: its W8 row-split tensors are the MTP and vision entries, which a
text-only greedy run with speculation off does not reach.

**On the output gate.** These runs cap generation at 32 tokens, and at that cap the five points
produce only four distinct completions, three of which coincide - so byte-comparing them, while it
passes 20 of 20, discriminates weakly. The gate that carries weight is a full generation: 33,031
tokens in, `--max-new 512`, which stops naturally at 164 generated tokens on the stop token. That
output is **byte-identical between the arms** - 203 bytes, an eight-field JSON answer block, one run
per arm. Since the change is bit-identical a greedy run is expected to match, so this confirms the
result in the product rather than establishing it.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host. Register and shared-memory figures
  come from `cuobjdump --dump-resource-usage` and instruction counts from `cuobjdump -sass` instead.
- The 38 concat instantiations are covered by the `linear_pair` measurement and the output gate, not
  by the register or instruction census.
- Repeat passes exist for `w8_linear_add`, `w8_linear_swiglu`, `linear_pair` and the two projection
  benchmarks. The `linear` suite, the `GROUPS == 4` sweeps and the routed-MoE control were each run
  once per campaign, so their cells carry the campaign-to-campaign spread and no narrower bound.
- Speculation is off in every end-to-end run, so the decode-shaped extents that the `GROUPS == 4` and
  projection sweeps cover at operator level are not exercised end to end.
- Qwen3.8-27B NVFP4 is not measured; it takes no W8 row-split route.
- The `nsys` attribution is at chunk 8192 only, four profile pairs.
- `--policy a8` is not measured separately; `select_w8_launch` maps `A16Only` and `AllowA8` to the
  same `select_w8_a16_launch`, so it reaches identical kernels.
- `docs/performance.md` publishes `qwen3_6_35b_a3b` prefill at a 1,024-token chunk with INT8 group-64
  KV, a path this change is on, so those figures would move. I have not re-run that harness; per the
  Issue, updating it is yours to schedule.
