# Decode and prefill optimization log — what was tried, how, why, and what it gave

Scope: every optimization with a measured decode or prefill effect found in the upstream
`Neroued/ninfer` history — merged, rejected, withdrawn and still-open work, ours and other people's —
plus the upstream commits that carry the mechanism. Baseline for all numbers unless stated otherwise:
**RTX 5090, `sm_120a`, 170 SMs, 525 W cap, CUDA 13.1.115, Release, Ubuntu 24.04**.

Purpose: when we bring up a new model or a new card, this is the list of levers that exist, the ones
that are already exhausted, and the ones that are known dead.

## Measured hardware ceilings used throughout (measure again on any new card)

| Ceiling | Value on this RTX 5090 |
|---|---|
| Sustained read (weights streaming) | **1689.4 GB/s** (also quoted 1674.5 GB/s in earlier runs) |
| DRAM copy (read+write) | **1503.4 GB/s** |
| L2 aggregate read | **7132–7461 GB/s** |
| L2 capacity | 96 MiB; `persistingL2CacheMaxSize` 60 MiB; `accessPolicyMaxWindowSize` 128 MiB |
| `mma_bf16` f32-accumulate | **253.4 TFLOP/s** (scalar tier 66.1 TFLOP/s) |
| FP8 tensor core | ~419 TFLOP/s peak; the project's FP8 GEMM reaches ~98% of it |
| NVFP4 MMA | **2003.9 TFLOP/s** measured (vendor spec quoted as 1736 TFLOP/s at base clock; ~2117 at 3105 MHz boost) |

Two structural facts that shape every decode result:

- Decode at T=1 is **latency- and launch-bound**, not bandwidth-bound. Typical decode kernels sit at
  30–45% of the read ceiling and issue no MMA at all. The levers are dependency chains, cold first
  touches, graph-node count and occupancy — not FLOPs.
- Prefill is **bandwidth- or MMA-bound** depending on width. The levers are traffic elimination,
  tile/rasterisation locality, pipeline depth and route selection.

---

# 1. Decode

## 1.1 Remove CUDA-graph nodes and kernel launches (fusion)

Every decode round replays a captured graph; each node carries fixed overhead (~1.4 µs of the
node-traced profile is instrumentation, but the real per-node cost is what these PRs harvest).

| Change | Mechanism | Gain | Status |
|---|---|---|---|
| Fuse Q/K RMSNorm + RoPE into one op for the text profile (D=256, (16,2)/(24,4), rotary 64) — PR #222 | third `ops::rmsnorm_rope` overload, out-of-place, bit-identical by construction (same sum-of-squares layout, same epilogue, BF16 round before rotation, same `fixed_sincos<Text1D>`); ~26 → ~24 nodes per round | **+0.596% decode** (8/8 passes, +0.445…+0.675%; zero control +0.021%), 35B-A3B | open |
| Fold the sigmoid gate into the causal-attention reduce epilogue — PR #225 | gate multiply moved into the reduce store for BF16/INT8 routes (65 of 320 cells); rounding order preserved so bytes match | operator **−8.33%** kernel time; **+0.311% decode** (10/10 passes) | open |
| Remove empty graph nodes of the MTP draft phase — PR #226 | `argmax` memset→initializer kernel node; MTP stem 3 nodes → fused `mtp_norm_pack_fc_input`; MTP tail 2 nodes → fused `mtp_residual_add_norm`; draft chain copy → two alternating buffers. One trace: 173 memsets, 397 rmsnorm, 134 pack, 129 residual, 82/100 D2D copies removed | **+0.334% decode**, 35B-A3B | open |
| Three redundant decode nodes (sigmoid gate, Q/K RMSNorm, MoE router selection folded into the last D1 block via `atomicInc` ticket) — PR #67 | node profile: `sigmoid_mul` 10×/step ~1.4 µs; Q+K norms 20×/step ~2.4 µs each; MoE D2 routing 40×/step ~4.8 µs of which only 1.8 µs was the selection | **+2.3% decode** on 35B-A3B (per-commit 1.004 / 1.012 / 1.006); +0.3% on dense 27B; +1.0% under MTP3 | closed → re-derived into #96, then split |

**Rule of thumb:** a fused pair is worth roughly 0.3–0.6% of decode per node class removed on the 35B
MoE target; the wins are small individually and only visible with a null-control arm.

**Negative result (#67):** folding RoPE into the norm kernel *was* implemented and measured — it
drifted by a last bit and cost ~1% MTP throughput through a **lower draft-acceptance rate**. Keep
RoPE in its own kernel. Bit drift in a speculative pipeline is not a rounding curiosity; it changes
acceptance.

## 1.2 Cross-kernel L2 prefetch (`prefetch.global.L2`)

The decode MoE chain leaves the memory system idle at two points — the D1/D2 routing window at the
head of the chain, and the D4 epilogue while the block waits on its barrier — and immediately after
each, a pure weight stream starts with a cold first touch. Between two MoE layers the model streams
~501 MiB through a 96 MiB L2, so nothing survives.

| Change | Mechanism | Gain | Status |
|---|---|---|---|
| Prefetch shared-expert `down` codes from the D1 tail — PR #203 | one `prefetch.global.L2` per thread over the `shared_down` code plane at 128 B stride, bounds-guarded by `kHidden*kIntermediate`; `launch_d1` takes `SparseMoeWeights` | operator T=1 cold **30.720 → 28.672 µs (−6.67%)**; E2E **+2.88…+3.28% decode** flat in context; MTP3 ≈ 0; prefill 0 | **merged** `ce954918` |
| Warm the next layer's projection from the D4 tail — PR #206 | `ops::SparseMoeHints` carried in `SparseMoeDecodePlan`; hint issued **before** `__syncthreads()`, one 128 B line per thread, clamped to 8 MiB | D4 itself **+6.4%** (18513.7 → 19694.4 ns) but the round **−1.8%**; E2E **+1.7…+1.9% decode**, flat in context | **merged** `7f14d963` |
| Both together, earlier form — PR #69 | same two hints | **+3.7% decode** (363.7 → 377.2 tok/s); consumer GEMV 580.6 → 482.2 µs (−98.4), issuing kernel +40.4 µs, step 3391.1 → 3305.6 µs | closed → resubmitted split |

Placement facts, all established by measurement and all load-bearing:

- **Issue the hint before the block barrier, not in the kernel tail.** In the tail it extends the grid;
  before the barrier it hides behind the slowest warp. Moving it cost **0.7 pp**.
- **8 MiB per layer is the measured optimum.** 12 MiB is 0.5 pp worse. The D1 window tolerates only
  ~1 MiB before it competes with D3's own stream.
- **Do not prefetch from a hot stream.** Issuing the hint from D3 — already at ~81% of achievable
  bandwidth — *lost* 1.1 pp. The gain exists only because those two windows are idle *and* the
  consumer is cold.
- The saving is **stall time, not bandwidth**: 1 MiB at 1689.4 GB/s costs 0.62 µs, while the measured
  saving is 2.048 µs (3.3× the peak DRAM time for those bytes).
- Graph-capture safe because the pointer is an ordinary kernel argument computed on the launch path.
  A thread-local slot was rejected on ownership grounds (Programs share no mutable state).
- Applies only to **T=1** (T≥2 takes `sparse_moe_small_t_launch`) and only to MoE artifacts.

## 1.3 L2 persistence reservation — currently a *negative* result

PR #202 pinned the Linear-Attention state pool (61.41 MiB read + 61.41 MiB written per decode round
across 30 layers) in L2: `cudaAccessPropertyPersisting` window installed **before**
`cudaStreamBeginCapture` (capture bakes the stream attribute into graph nodes and never re-reads it),
`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 60 MiB)`, `hitRatio` clamped to the fraction that
physically fits (0.4885), and the whole mechanism disabled when the pool exceeds
`accessPolicyMaxWindowSize`.

Physics were convincing: `recurrent_fold_kernel` 54 944 → 41 216 ns (**−25.0%**), 2343.8 → 3124.5 GB/s
= 2.08× the DRAM copy ceiling and 92.5% of the L2 read ceiling. Decode +0.5…+0.8%, prefill −0.709%.

It was withdrawn because on a base that already contained #203 and #206 the sign inverted: **decode
−0.217% (10/10 negative), prefill −1.036% (8/8 negative)**.

> **The transferable lesson:** L2 capacity is a single shared budget. Reserving a slice for one
> consumer takes it away from every prefetch that another merged change relies on. Two independently
> validated L2 optimizations can cancel. Any persistence work must be re-validated on the base of the
> day, against the other L2 users that exist at that moment.

Also measured there: taking the reserve *without* pinning costs the same kernel +34.5% — the carve-out
removes L2 the kernel was already getting opportunistically.

## 1.4 Occupancy-aware prefetch inside a kernel

PR #150 / commit `9954867a` — **RMSNorm reads `weight` (and `z`) before the reduction.** The kernels
made two sequential trips to memory per row; neither `weight` nor `z` depends on the sum, so the second
trip is issued inside the first and the two overlap instead of running back-to-back across the barrier.
Output is bitwise identical (reduction loop untouched, epilogue reads the same pairs out of registers).

- Kernel: e.g. `target_hidden35` T=64 1727 → 1152 ns (**0.667**); T=1024 3104 → 2592 ns (0.835).
  Norm pool −33.2% (61.01 → 40.75 ms); whole run −1.09%.
- End-to-end: **27B decode +2.18% (no spec), +1.49…1.64% (MTP3); 35B-A3B +2.10…2.63% (MTP3)**;
  prefill unaffected.
- The hoist costs registers (gated warp 35 → 50, gated wide-row 38 → 48) and drops 512-thread blocks
  from 3 to 2 per SM, so it is a **template parameter with no default** — every launch site states its
  choice — and it is additionally gated on a grid of ≥ **170 blocks** (one per SM on this part).
  That constant is this card's SM count written as a literal: **re-derive it on any new GPU.**
- nsys context that justified the work: *"rmsnorm costs 3% of decode wall time"*.

## 1.5 Restructuring a serialized kernel (grid + reduction network)

PR #191 / commit `487f8977` — **small-T MoE S2**: `launch_s2` wrote `dim3(1)` in both branches, so
every one of 1240 batch-1 and 1200 batch-8 decode launches ran on a **single CTA** while S1 of the same
operation ran on 1028 blocks. Inside the warp, top-8 selection walked eight dependent
`sparse_moe_warp_best` rounds (~48 dependent shuffle levels) although every lane already held its eight
candidates sorted.

Two edits, both needed: one CTA per token (`dim3(tokens)`, 128 threads; shared memory becomes 2 084 B at
any T instead of 2 084 + 1 060·(T−1) up to 47 664 B), and a five-level `__shfl_xor` merge network
replacing the dependent chain.

- Kernel: T=4 5 568 → 4 000 ns (**−28.2%**); T=32 12 608 → 8 608 ns (**−31.7%**); across T=2..46,
  −22.4% … −51.5%.
- Round: −1.111% … −2.743%. **End-to-end decode +1.598%** (658.99 → 669.51 tok/s), re-measured
  +1.469% on merged master.
- Ablation: CTA-per-token alone −0.226% (T=4) / −1.380% (T=32); merge alone −0.936% / −0.428%.
- Cost: prefill **−0.308%** (10/10 negative) because `sparse_moe_prefill_select_count` went 40 → 54
  registers. Reported as a correction after merge.
- The merge network is hardcoded for **256 experts × top-8** — a different expert count needs a
  different network.

Kernel was 0.175–0.905% of the read ceiling and 0.0012–0.0043% of the scalar tier: **when a kernel is
that far from every ceiling, the bottleneck is the dependency chain or the grid, and the fix is
structural.**

## 1.6 Memory-level parallelism inside a dot product

PR #199 — **keep two Q4 group quads in flight** in the routed gate/up dot product (`dot_two_rows`,
`sparse_moe_decode_kernels.cu`). The loop issued one quad's codes, scaled and decoded immediately,
leaving the load unit idle during the decode and its 16 FMAs. Issuing both quads of a pair before either
is decoded lets the first quad's decode/FMA chain cover the second's memory latency. Output bit-identical.

- Operator (q4-q5, T=4, cold): −4.78% (`same`, 8 experts), **−7.40%** (`trace-like`, 22), −6.06%
  (`independent`, 31). Round kernel median −2.75%; kernel total −3.11%.
- End-to-end MTP3 **+0.57%** (413.61 → 415.99 tok/s), re-measured **+0.524%** (9/10 passes).
- **Regression caught in review:** the shared helper also instantiates
  `sparse_moe_d3_nine_warp_kernel<Q4Codec>`, the T=1 non-speculative path that runs for *every*
  generated token — it slowed **6.1–6.6%** (later measured +10–11%). Fixed by giving callers separate
  `QuadsInFlight` depths (nine-warp caller passes 1); SASS then byte-identical to master for that body.
- The path is at 55.1% of the read ceiling with zero HMMA/IMMA/OMMA: **latency, not bandwidth.**

## 1.7 Schedule/threshold retuning (and why it is fragile)

PR #200 — widen the Q5 routed-down `Rows2` window. `Rows` is a grid-split parameter only
(`grid = dim3(kHidden / Rows, tokens)`); `Rows2` doubles the CTA count versus `Rows4`, giving twice the
parallelism to hide DRAM latency while the grid is small, at the cost of re-reading per-expert
activation slabs in every block (+63% L1 traffic; L2 and DRAM flat). The crossover is a
*latency-hiding vs activation-re-read* trade.

- Operator: Q5 T=12..19 **+1.23…+1.82%**, Q6 T=12..16 +1.33…+3.51%.
- **Rejected**: at Q6 T=17 under concentrated routing (`--codec q4-q6 --distribution same`) the
  candidate was 129.056 µs vs baseline 127.008 µs = **+1.6125% Op-time regression**, with the identical
  baseline control at exactly 127.008 µs.
- Post-mortem: the crossover is **machine-dependent through SM clock**. Q5/Q6 crossovers are 17/11 on a
  1975 MHz host and 19/16 on a 2810 MHz host; the lower-clocked machine binds because the cost of
  `Rows2` is SM-side. No single Q6 threshold holds across both.
- The crossover also **moved between bases** (T=16 → T=19) after master rewrote
  `sparse_moe_small_t_kernels.cu`.

> Threshold constants are the least portable thing in the tree. On a new card, every crossover in
> `*_plan.cpp` must be re-swept, and the binding case is the *slowest-clock* machine, not the fastest.

PR #201 — **choose the predicated W8 GEMM cache policy instead of inheriting it.** The predicated
(ragged) activation copy inherited `cp_async_zfill`'s `Cache::ca`. Mechanism knowledge worth keeping:
at BM=16 twice as many CTAs read the same activation tile (320 vs 160 on the feature route), so
`Cache::ca` keeps it in L1 where co-resident CTAs hit it; forcing `Cache::cg` drops the L1 sector hit
rate **35.61% → 0.64%** and multiplies L2 traffic by **×1.549** (matching the 1/(1−hit) prediction of
1.553). At BM≥32 the balance reverses. Deferred by the maintainer over an unresolved +0.475…+1.037%
residual on the BM=16 feature route at T=60/64.

## 1.8 Draft head / speculation levers

- **Round speed vs round count.** Every accepted decode PR in this corpus separates the two by showing
  `spec_rounds` / `spec_acceptance_rate` identical across arms (e.g. 458 cells with `spec_rounds` 678
  and acceptance 0.7994100295 in every cell, #202). If acceptance moves, the throughput number is not a
  kernel result. Aggregate committed throughput is `round speed × acceptance`, and the published
  Qwen3.8-27B NVFP4 profile is a live example: 45.8–48.9% acceptance versus 67.2–71.4% elsewhere.
- **Full-vocabulary Q4G64 MTP proposal head** (issue #174, open, not answered): requantize the FP8
  `lm_head` on-device into row-split Q4G64 keeping all 248 320 rows. FP8 GEMV is bandwidth-bound
  (752 µs, 1.272 GB, 1.69 TB/s ≈ peak), Q4 full-vocab is 402 µs / 0.675 GB — **~47% fewer bytes, ~47%
  faster**. End-to-end +8.3…+9.0% decode with accuracy preserved (structured JSON 99.0% vs the
  truncated head's 86.0%). Costs 0.63 GiB resident, which makes it unusable at 490K context on 32 GB
  (paging: prefill −33%). **Strong candidate on a bigger card.**
- Truncating the draft head to 131 072 rows is what costs the JSON acceptance (99.0% → 86.0%).
- MTP graph profiles must carry a topology class (PR #221): `cudaGraphExecUpdate` cannot cross a change
  of node count, so profiles that span the SmallT↔chunked attention route flip share one
  `cudaGraphExec_t` and fail with `cudaErrorGraphExecUpdateFailure` for draft windows ≥ 6. Fixing this is
  a precondition for ever raising `kMtpDecodeMaximumDrafts` above 5. (Contract violations 1150 → 0 over
  40 (capacity, k) pairs; no-op at the shipped cap.)

## 1.9 Third-party result worth reproducing: W8 short-K decode pipeline depth

Issue #216 (`ranxianglei`, adopted): W8A16 o_proj/fc1-class GEMMs at K=6144 and K=5120 in decode ran at
22–35% memory-bandwidth efficiency because `STAGES=2` gives only **one tile of lookahead** — per-iteration
compute ≈350 ns against ≈800 ns of DRAM wait, leaving ~450 ns exposed per iteration (96 × 800 ns = 76 µs
against an ideal 23.7 µs → 31%, matching measurement). `R64C128` also produced only 80 CTAs. Switching to
`BK=128 K128A1` schedules took **Qwen3.8-27B W8A16 C20 aggregate decode from ~490 to 1080 tok/s** on an
RTX 5090 — parity with SGLang/marlin (~1086). Related correctness fix in #215: the split-K launchers
hardcode `kRows = 2048` in three places and silently corrupt any output with rows ≠ 2048.

---

# 2. Prefill

## 2.1 Rasterisation and tile locality (the biggest single prefill lever found)

PR #204 / commit `ee9d5192` — **rasterise the W4A4 TMA CTAs token-fastest, and fetch each
activation-scale box once.**

- The grid was `(kOutputRows / kBlockN, tokens / kBlockM)` with `blockIdx.x` fastest, so concurrent CTAs
  never shared a weight tile and the weight matrix was re-read **once per token tile** (16× at T=4096).
  Decoding both tile indices from the linear CTA id with the **token index fastest** makes CTAs that
  share a weight tile run together — the same `Bf16MmaRaster::TokenFast` choice `bf16_gemm_mma_kernel`
  already made.
- TMA cannot address a box narrower than 16 B, and 16 B of scales cover two K tiles, so the load at
  `(k_tile/2)*16` was issued **twice for the same bytes**; now issued on the even tile only into a
  two-slot buffer, with the odd tile's barrier expecting `kTransactionBytes - kScaleBytes`.
- Result: a K tile moves 27 648 B (generic) / 28 672 B (SwiGLU) — the exact minimum for
  BlockM=256, BlockN=128, K=128. At T=8192 tile traffic 10.70 → 9.98 GB (**−6.7%**);
  **1014 → 1268 TFLOP/s = 50.6% → 63.3% of the measured NVFP4 MMA tier.**
- Operator ×1.09…×1.20 (T=256…8192). **End-to-end Qwen3.6-27B NVFP4 prefill +9.29% / +8.13% / +7.65%**
  at prompt 1024 / 4096 / 8192 (chunk 4096). Prefill-heavy mix **+19.4% req/s**. Registers, shared
  memory, OMMA and LDSM counts all unchanged — the growth is index arithmetic (+5.5% SASS).

**Generalisable:** on any new GEMM route, check (a) which CTA index varies fastest relative to the
operand you want resident, and (b) whether the TMA box granularity is making you fetch the same bytes
more than once.

## 2.2 TMA staging with a producer warp, bigger tiles, deeper narrower pipeline

PR #167 (open, four maintainer items resolved) — **the FP8 A8 GEMM stages its operands through TMA.**

| | cp.async route | TMA route |
|---|---|---|
| Output tile per CTA | 64×128 | **256×128** |
| Consumer warp tile | 32×32 | **64×64** |
| Accumulator registers / consumer thread | 32 | **128** |
| K pipeline | 2 stages of K=128 | **4 stages of K=64** |
| Threads per CTA | 256 (8 warps) | **288** (8 consumer + 1 producer) |
| CTAs per SM | 2 | **1** |
| Shared memory | 100 352 B | 98 816 B |
| Registers | 94 | 166 |

Why it works: the wider warp tile makes each `ldmatrix` of A feed eight N fragments instead of four —
MMA issued per operand load goes **8 → 32** — and the deeper, narrower K pipeline with a dedicated
producer keeps that tile fed without per-stage CTA-wide barriers.

- Operator ratio 0.877–0.989 (`linear`), 0.895–0.979 (`linear_add`).
- **End-to-end prefill on Qwen3.8-27B: +2.40% at chunk 1024, +4.95% at 4096, +4.38% at 8192.**
  Decode unchanged (kernel never launched in the decode graph).
- Limits: 9 warps/SM instead of 16 means narrow shapes are slower — there is a width floor, and the
  0.936 wave-count admission constant is **fitted for this part and is not portable**.
- Latent defect found on the way: the 64-byte swizzle pattern repeats at a **512-byte** boundary, but
  the shared storage only declared 128-byte alignment; CUDA 13.1 happened to lay it out safely.
  `static_assert` the real precondition.

## 2.3 Traffic elimination

| Change | Mechanism | Gain | Status |
|---|---|---|---|
| Stage routed MoE gate/up from `x` instead of a gathered copy — PR #140 / issue #117 / `0a204aee` | `sparse_moe_prefill_gather_kernel` materialised one 2048-wide BF16 row per (token, expert): at top-8 of 256 a 4096-token slice wrote 32 768 rows × 4 KiB = 128 MiB, read straight back. It is replaced by an index kernel publishing a `packed_token` map; the staging loop resolves each packed column's source row once per route job into `src_row[]` in **registers** and walks pointers into `x`. The map aliases the dead prefix of `tile_counts` | gather kernel 55.59 → 0.00 ms; operator ×1.038–1.053 (T=1024–8192); **E2E prefill +1.38…+1.86%**; 256 MiB less traffic per 4096-token slice (~37% of combined traffic); compute efficiency 46% → 48% of the tier | **merged** |
| Raise `kSparseMoePrefillSliceMax` 4096 → 8192 — issue #118 | an 8192-token chunk was processed as two independent 4096 passes, so every expert's weight slab was read twice. Nothing in the kernels required 4096; it only bounded workspace | operator at T=8192: q4-q5 −6.8%, w8-w8 −8.1%; traffic 231.5 → 123.8 GB/s and 345.4 → 187.4 GB/s; **E2E +2.06% at chunk 8192** | adopted |
| GDN prefill convolution writes q/k/v directly — PR #99 / `92bb06eb` | removes the intermediate `[C,T]` packed plane and three `cudaMemcpy2DAsync` extractions per layer per chunk (32 GB of copies per 32K prompt) via `CausalConvSplitOutput3` with **compile-time** partition boundaries | split vs packed+extract ×1.68–3.50 (T=17–64), ×2.49–2.50 (T≤16), 360.48 → 206.85 µs at T=8192; **E2E prefill +0.93…+1.34%**; workspace **963 → 835 MiB** at chunk 8192 | **merged** |
| Widen W8 row-split weight decode to eight codes per thread — PR #112 / `93cdc264` | staging re-indexed from per-row-pair (warp) to per-thread over eight-code chunks: a thread owns a whole chunk inside one quantisation group, reads its own scale halfword, decodes eight codes from one `uint2`, stores 16 B once. Removes all three `shfl_sync` broadcast sites | LDS −48.4%, STS −40.2%, SHFL 1150 → 0; changed kernel ×1.123; production GEMM median ×1.089; **E2E prefill +2.46% (8 515 tok) / +1.55% (65 882 tok)** on 35B-A3B; dense 27B null | **merged** |
| Same treatment for MoE prefill staging — PR #106 / `8d343527` | `decode_eight` atoms for Q4/Q5/Q6 | STS/LDS −75% on `<Q5DownMma,8,64>`; operator ×1.05 (T=1024) → ×1.10 (T=4096–8192) for q4-q5 and q4-q6; w8-w8 unchanged | **merged** |

## 2.4 Route/predicate coverage — free gains from letting a good kernel run

- `00369f63` **register the fused TMA SwiGLU for every width its block accepts.**
  `nvfp4_linear_swiglu` registered the fused route at exactly T == 1024 although the kernel accepts any
  multiple of its 256-token block; every other width fell back to `LinearW4A4Post`, which writes a
  34816×T bf16 tensor to the arena and reads it back through a separate `silu_mul`. Moving the predicate
  to "every multiple of 256" covers every prefill chunk width the product uses. The workspace capacity
  function had to move with it, or the shipped Op test reports a query/execution high-water mismatch —
  PR #115 added the T=256 oracle case that catches exactly that.
- PR #198 / `437e9f98` **choose the routed gate/up pipeline depth by route, size the grid by work.**
  Three coordinated changes as one mechanism: `cp_async` stages 2 → **6** in the narrow routed gate/up
  (six stages fit the 48 KiB static shared limit exactly: 8192 + 6·4096 + 6·2048 + 4096 = 49 152 B);
  the grid moves from a fixed `3 × 170 = 510` blocks to one sized from the actual route job count
  (capped at 32 blocks/SM); and both depths are instantiated and launched over the same work list with a
  **device-side** predicate choosing between them — *deep while `4·jobs < 7·touched`*.
  - Why depth matters: a job is one non-empty 32-column tile of one expert. More than one job per touched
    expert means neighbouring jobs re-read weights already resident in L2 — there is no latency left to
    hide, and the deep pipeline just pays 24 KiB more shared memory, dropping resident blocks/SM from 3
    to 2.
  - Roofline that identified the target: at 256 tokens `q4_gate_up` ran at 649.8 GB/s = **38.5%** of the
    read ceiling while `qx_down` of the same operation ran at 1445.0 GB/s = **85.5%** — a 2.2× gap
    between two kernels of one op. Six stages take the gate/up to 57.3%.
  - Operator −16.3% (128 tok) … −5.5% (4096 tok), and the 8-expert concentrated case goes from a +6.4%
    regression to −2.4%. **End-to-end server prefill −2.25% weighted mean over 23 length bins**, best in
    the 399–746-token band where median layer calls sit at 1.75–2.0 jobs/expert.
  - Warm-L2 and cold-L2 give non-overlapping crossing ranges; **the product matches the cold side.**
- PR #194 (open) **drop the guarded `expf` slow path from the fused NVFP4 SwiGLU epilogue** —
  use the hardware-approximate `__expf` instead of the accurate `expf` plus its range guard inside
  `nvfp4_linear_swiglu_w4a4_tma_kernel`. Ablation: **5.73% at T=8192**. Review caught the real hazard:
  the naive `__fdividef` form of SiLU returns exactly zero over part of the input range, so the fix has
  to avoid that range while keeping the fast path. Qualified against the FP64 oracle plus perplexity.
  A reminder that transcendental fast-math in an epilogue is a genuine lever on a bandwidth-bound
  GEMM — and that its failure mode is a silent zero, not a small error.
- PR #160 (open) **NVFP4 activation-scale plane, row-major → blocked.** The TMA engine delivered
  `BlockM` separate 16-byte transfers for the scale plane: 73% more time for 16.7% more bytes. Blocking
  into one `[BlockM tokens, 16 groups]` tile restored codes-only throughput (**4732 → 7009 GB/s**),
  operator 0.806–0.865, **prefill +4.09%** (9426.4 → 9811.8 tok/s). **Superseded:** re-measured after
  `ee9d5192` it regresses 6.0–9.9% at every T — #204 already collected the same bytes. Keep as a
  worked example of TMA box granularity, not as a patch.

## 2.5 The long-context ceiling

Issue #32: at ~260K context NInfer (5157.1 tok/s) and vLLM NVFP4 (5124 tok/s) land **0.6% apart**, while
at short context they differ by ~37%. The maintainer's conclusion: *"Yes, attention is the bottleneck at
long context. But quantizing attention computation needs to be done carefully."* Weight-format work does
not move the long-context prefill number; attention does.

The one attempt on that ceiling was **rejected**: issue #27 proposed fp16-accumulate PV tiles in the INT8
attention prefill route (GB202 runs f32-acc HMMA at half the fp16-acc rate), with real numbers —
185.9 → 226.3 TFLOP/s at 8K, 192.5 → 229.6 at 128K, end-to-end 1607 → 1682 tok/s at 200K, exact needle
retrieval at 64K/128K/200K. Rejected on precision: *"In attention, PV is much more precision sensitive
than QK, even use bf16 PV with fp32 acc is worse than fp16 PV with fp32 acc. So using fp16 acc is no an
option."*

---

# 3. Host side and TTFT (cheap, often ignored)

| Change | Mechanism | Gain | Status |
|---|---|---|---|
| Skip NFC normalisation for pure ASCII — PR #205 / `641ef3e7` | one raw-byte scan; every byte < 0x80 is a starter with ccc = 0, so ASCII is already NFC. Malformed UTF-8 exits on the first high byte | `normalize_nfc` ×46–×71; `Tokenizer::encode` **−13…−22%**; with a prefix-cache hit tokenisation is **79–81% of TTFT** and TTFT drops **4.9–14.8%**; cold long prompts: no measurable change | **merged** |
| BPE merge rules in a flat open-addressed table — PR #193 / `b158afe2` | `unordered_map<uint64,BpeMergeRule>` (bucket array + one heap node per rule) → power-of-two array of 16-byte entries with linear probing at 2× the rule count. Rules never change after load and their count is known before the first insert. **Rank order is load-bearing:** with linear probing the first claimant keeps its slot, so walking `model.merges` in rank order gives frequent merges their home slot — filling the same table out of an unordered container costs **10.7%** of encode | `encode` 0.746–0.804 of base over 45 cells (median 0.764); resident memory 9.51 → 8.00 MiB table, live heap −1.516 MiB; worth **0.95–1.05% of a 64-token turn and 6.3–9.5% of TTFT** | **merged** |
| Prefix-cache *eligibility*, not capacity — issue #62 | a prompt with a stable head and a volatile tail (timestamp, sensor text) diverges **before** the checkpoint, so nothing is ever restorable; more KV capacity does not help. Measured 121 of 124 requests `full_reset`, 984 ms/turn vs vLLM 209 ms. A separate variant: the checkpoint anchors at the start of a tool loop and never advances (71 of 439 real agent requests took 30–36 s TTFT) | resolved by host-backed parking (#51/#73) plus checkpoint work | closed |
| Materialization search budget — issues #176, #229 | `search_budget_ns = min(5 ms, incumbent.cost.total_ns / 20)`: the `/20` never binds, so the flat 5 ms cap verified only ~10 candidates at 0.3–0.7 ms each and a 130 568-token prompt whose prefix was fully reusable got `cached_tokens = 0` and a 142.6 s TTFT. Widening to `min(250 ms, max(5 ms, cost/20))` gave 130 535 cached tokens and **1.2 s TTFT** | maintainer replaced it with a broader design in `d4929686` (construct promising complete candidates first, validate/repair with exact feasibility checks, start short and extend the budget when expected remaining benefit justifies the completion cost, accounting for other runnable requests) | fixed upstream |

---

# 3.5 KV cache formats and the cost model

- **Compressed KV**: `danielfparkernz` proposed an E8-lattice compressed KV for the Blackwell
  `sm_120a` path (PR #35, closed *"Superseded by `4ac73c4`"*), and `splickz` proposed `rk4v4-e8`
  for SM120 (issue #123). The maintainer published an experimental branch
  (*"Here's experimental branch which supports nvfp4 kv and fp8 k + nvfp4 v (k8v4), give it a try"*)
  and then landed it himself in `4ac73c4`. A follow-up `rk2v4-e8` PR (#173) is still open.
  Pattern: KV-format work is welcome as a *proposal with numbers*; the implementation tends to be
  done upstream.
- **KV precision tail** (issue #164, open, no maintainer answer): keep the most recent N tokens at
  higher precision than the rest, as llama.cpp's `--kv-tail-tokens`. No measurements were attached,
  which is probably why it got no traction.
- **The prefill cost model is keyed on (model, weights) and is only populated on a diagonal** —
  PR #195. When a `(model, weights)` pair has no compiled row, the planner falls back to a generic
  profile that is 2.78× too slow, and the measured prediction error was **3.08–3.15×**; consulting a
  preset of the *same weights format* after the exact pair misses brings it to **1.15×**. This is a
  direct hazard when registering a new artifact or a new card: the planner will mis-predict prefill
  cost long before anyone notices a throughput problem. Related: issue #175 — an RTX 3090 running on
  the 5090 cost profile produces prefill prediction errors.

---

# 4. Dead ends and traps (do not spend time here again)

1. **fp16 accumulate for PV in attention** — rejected on precision; even bf16 PV with fp32 acc is worse
   than fp16 PV with fp32 acc (#27).
2. **Reserving persistent L2 for one consumer** while other kernels prefetch into L2 — net negative
   (#202).
3. **Prefetch hints issued from a bandwidth-saturated kernel** — −1.1 pp (#69).
4. **Prefetch in the kernel tail instead of before the barrier** — −0.7 pp (#69).
5. **Prefetch caps above the measured optimum** — 12 MiB is 0.5 pp worse than 8 MiB; >1 MiB in the D1
   window competes with D3 (#69).
6. **Fusing RoPE into the norm kernel** — last-bit drift, ~1% MTP loss through acceptance (#67).
7. **Applying a widened pipeline/depth through a shared helper** without per-caller depth — regresses the
   T=1 nine-warp decode kernel 6.1–11% (#199).
8. **Extending a measured crossover past the binding routing distribution** — +1.61% Op regression at
   Q6 T=17, and the crossover itself moves with SM clock and with the base (#200).
9. **`Cache::cg` for predicated activation copies at BM=16** — L1 hit rate 35.61% → 0.64%, ×1.549 L2
   traffic (#201).
10. **Keeping the map in shared memory, or recomputing it inline**, instead of in registers — 0.4–1.4%
    slower (#140).
11. **A thread-local or module-global channel** for passing hints — rejected on ownership grounds even
    when it measures the same (#69, #96).
12. **Runtime-general kernels for registered geometries** — the split convolution paid 3.2–5.0% for
    generality it did not need (#99).
13. **Raising the prefill chunk default to buy prefill throughput** — workspace goes 120 MiB (1024) →
    482 MiB (4096) → 963 MiB (8192) straight out of KV/context capacity (#96).
14. **Trusting an operator fixture as the decision number** — #199's fixture overstated by 2.69×; the
    round-level delta was the decision-relevant figure. And in #198 the "trace-like" fixture routing did
    not match the model's real routing, which had to be instrumented over 29 token counts.

---

# 5. Measurement practice that made results defensible

- **Null control arm**: the identical baseline compiled and run as a third arm; it must read ~0.000%.
  Every accepted PR here reports it (#200's control read exactly 127.008 µs against a 127.008 µs
  baseline).
- **Same process, interleaved orders**: all candidate graphs captured in one process over shared
  inputs/buffers, replayed through every arm order, 20 warmups + 300 samples per arm, 256 MiB
  cold-cache flush before each call. Separate-binary comparisons are corroboration, not the primary
  evidence.
- **In-band clock witness**: a fixed dependent-ALU chain that touches no memory, timed with every
  repeat; repeats whose witness is >3% off the best of the run are dropped. Without it, a host's three
  P-states widen the floor to about ±10% (#193).
- **Instrument floor**: a byte-identical baseline arm on both sides, plus builds with a dead function
  ahead of the hot code to shift it by e.g. 1438 and 8901 bytes — the floor must be measured, not
  assumed (#193's floor was 105 cells at 0.941–1.021 and the worst mechanism cell cleared the best
  floor cell by 13.7 points).
- **Sabotage / strength control**: prove the identity gate can fail (deliberately damaged builds), or a
  "bit-identical" claim is worthless.
- **Resource census** for every changed body: registers, shared, stack, local, spills, constant bank,
  and the SASS/HMMA stream. "MMA stream unchanged, what shrinks is the work around it" is the standard
  argument for a staging optimization.
- **Speculation counters** (`spec_rounds`, `spec_fallback_steps`, `spec_acceptance_rate`) to separate
  round speed from round count.
- **nsys node trace inflates absolute values** (~1.4 µs/node); deltas are usable, absolutes are not.
- `ncu` is frequently unavailable (`RmProfilingAdminOnly`), so bandwidth claims are often *demand*
  derived from bytes moved rather than counters — say which, and do not assert DRAM traffic from
  `lts__t_bytes` alone.

---

# 6. Checklist for a new model or a new GPU

**Re-derive these constants — they are this card and this artifact, not physics:**

- SM count literal used as the RMSNorm prefetch grid gate (**170**).
- The 8 MiB L2 prefetch cap and the ~1 MiB D1-window budget.
- Every crossover in `sparse_moe_small_t_plan.cpp` and friends (Q5/Q6 `Rows2` windows, small-T/prefill
  route boundaries) — sweep on the **slowest-clock** machine you must support.
- The routed gate/up depth predicate `4·jobs < 7·touched`, and the 6-stage shared-memory budget
  (49 152 B against the 48 KiB static limit).
- The FP8 TMA admission model, especially the fitted **0.936** wave-count constant and the width floor.
- Fused-route width predicates (multiples of 256 for NVFP4 TMA SwiGLU; MTP fused widths 2048/5120).
- The measured ceilings table at the top of this document.
- **The compiled context-cost table**: a new `(model, weights)` pair with no row falls back to a
  generic profile that was measured 2.78× too slow, giving a 3.1× prefill-cost prediction error
  (PR #195). Add the row, or at least the same-weights-format fallback, when registering an artifact
  or moving to another card.

**Re-check these structural assumptions:**

- L2 size versus per-layer streamed bytes (96 MiB vs ~501 MiB between two MoE layers here) — the whole
  cross-kernel prefetch family depends on the consumer being cold.
- Expert count and top-k (the S2 merge network is hardcoded for 256 experts × top-8).
- Whether the decode path is MoE at all (the dense 27B never reaches `sparse_moe_d1/d4`, so several
  merged decode gains are structurally zero there).
- Weight-format mix: the Qwen3.8 NVFP4 artifact reads ~17.8 GB of weights per decode step against
  ~14.3 GB for the Qwen3.6 NVFP4 artifact — ~20% of the decode difference is bytes, before any kernel
  work.
- MTP acceptance for the new checkpoint. Aggregate throughput is round speed × acceptance, and
  acceptance is a property of the model and the sampling parameters (the default temperature change in
  `33627e8` lowered it deliberately).
- Attention accumulate is the long-context prefill ceiling; do not expect weight-format work to move it.

**First places to look for headroom on a new target,** ordered by what paid off here:

1. Kernels far from every ceiling (<10% of read *and* of the compute tier) — those are dependency-chain
   or grid problems (#191 gave +1.6% decode from one such kernel).
2. Two kernels of the same operation at very different fractions of the read ceiling (#198: 38.5% vs
   85.5% → +2.25% prefill).
3. CTA rasterisation order versus the operand you want resident, and TMA box granularity (#204: +8%
   prefill).
4. Cold first touches immediately after an idle memory window (#203, #206: +1.7…+3.3% decode).
5. Graph-node count per decode round (#67, #222, #225, #226: 0.3–2.3% each).
6. Pipeline depth versus exposed DRAM latency on short-K decode GEMMs (#216: 490 → 1080 tok/s).
7. Host-side TTFT: tokenizer and normalisation are 79–81% of TTFT on a prefix-cache hit (#205, #193).

---

# 7. Index of upstream perf work by theme (for archaeology)

Themes visible in upstream's own `perf(...)` commits (~276 of them), useful when checking whether an
idea has already been tried:

- **Fusion**: `ad17bc62` fuse hidden norm into control projection; `7ad7966c` fuse qk normalization into
  recurrence; `3b6c32cb` fuse small-t input projection snapshots; `02be37cb` fuse and qualify
  variable-width gdn norm control; `1c8f8acc` fuse nvfp4 swiglu through t96; `0f0899c9` extend fused
  target swiglu.
- **Launch overlap / PDL**: `3a3205c2` sparse-moe PDL launch overlap; `7c543611` overlap snapshot
  projections with PDL; `35952722` reduce CUDA graph startup work.
- **Staging / decode width**: `93cdc264`, `8d343527` (eight codes per thread); `0a204aee` stage from x.
- **Route/tier selection**: `0ba16ed9` select measured nvfp4 activation routes; `aa429f28` route gdn
  input t16 through grouped mma; `4b0eb36c` route h24 verify attention through small t; `22d8a1d3` w8
  vocabulary t64 route; `6d1da9ce` remove q5 linear add aggregate cliff; `2fb9d830` sparse-moe
  tensor-core prefill path; `3a440960` small-t simt path; `2a753f5e` extend small-t persistent path to
  t32.
- **New quantised kernels**: `5522a878` nvfp4 w4a4 mma projection; `a3896165` bf16 attention prefill
  mma; `1fc1cb76` fp8 w8a16 vocabulary gemm; the `w8 …` family (`f8afbb7b`, `61b09bac`, `1c7082fa`,
  `329e81c8`, `9404de27`).
- **DFlash2 wave** (2026-09-04…09-06): `377f71a1`, `426d129f`, `221baf55`, `95037fe4` plus a long tail of
  `perf(ops): qualify/tune variable-width …` commits — the pattern for bringing a new speculative
  backend up to speed is: add the op, qualify it, then retune every variable-width route around it.
- **Runtime/host**: `0e6d4f45` index prefix checkpoints by content; `8554dfa0` partition host kv extents
  by run; `268409bf` batch page allocation and mapping; `8f0aea6f` commit only changed kv frontier
  pages; `02bd904e` keep mtp prefill token on device; `604bdc5f` optimize thinking-preserving prefix
  reuse; `d4929686` materialization search and adaptive planning budgets.
- **Vision/media**: `fc5c4834` reclaim transient workspace; `3b607545` streamline multimodal preparation;
  `9e45f27d` uniform segment attention.

## Our contributions that landed upstream (14 commits)

`92bb06eb` (#99), `8d343527` (#106), `93cdc264` (#112), `00369f63` (fused TMA SwiGLU width predicate)
+ `5327d676` (#115, the oracle case that guards it),
`0a204aee` (#140), `9954867a` (#150), `487f8977` (#191), `b158afe2` (#193), `437e9f98` (#198),
`ee9d5192` (#204), `641ef3e7` (#205), `ce954918` (#203), `7f14d963` (#206).

Still open at the time of writing: #160, #167, #194, #195, #199, #201, #221, #222, #225, #226.
Closed without merge: #67, #69, #96, #200, #202, #220, #224.
