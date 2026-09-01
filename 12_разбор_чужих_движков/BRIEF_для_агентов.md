# Brief for study agents: what NInfer is, and what NOT to propose

You are studying a THIRD-PARTY engine's source to find mechanisms NInfer could borrow.
Read this first. Proposals that violate it are wasted work.

## What NInfer is

`https://github.com/Neroued/ninfer`. Local read-only clone at
`C:/Users/MixaPC/AppData/Local/Temp/claude/corpus/ninfer` (upstream master `da49c0d`).
C++/CUDA, AOT-compiled, no Python at runtime, no PyTorch, no Triton, no JIT.
Targets **one GPU only: RTX 5090, `sm_120a`**. Single user; concurrency 1..8, usually 1.
241 `.cu`/`.cuh` files. Layout: `src/ops/{kernel,launcher,linear,linear_swiglu,
softmax_attention,linear_attention/gated_delta_net,sparse_moe/{prefill,decode,small_t},
kv_cache,...}`, engine in `src/runtime`, `src/serve`.

**Therefore: anything that is multi-GPU (TP / PP / EP / DP, all-reduce, NCCL, NVLink,
pipeline stages), anything that needs Hopper/B200 (`sm_90a`, `sm_100a`), and anything whose
value is "a Python-level abstraction" is OUT OF SCOPE.** Do not report it.
The parallelism findings in the benchmark report exist only to explain *why* a number moved;
the transferable part is always the single-card mechanism underneath.

## Models NInfer runs

- **Qwen3.6-35B-A3B** — 40 layers = **30 GDN (gated delta net, linear attention) + 10 GQA**;
  d_model 2048; MoE 256 routed + 1 shared expert, top-8, d_ff 512; 16 Q heads / 2 KV heads,
  **head_dim 256**, RoPE applied to only 64 of 256 dims; vocab 248077; MTP draft head 131072.
  Weight bytes/token 2.52 GB, GDN state 120 MB, KV S x 10.56 KB.
- **Qwen3.8-27B (NVFP4)** — 64 layers = 16 full attention + 48 GDN; hidden 5120;
  24 Q heads / 4 KV heads head_dim 256. MLP layers 0-55 in **NVFP4 block-scaled**
  (`mma.kind::mxf4nvf4`, m16n8k64); everything else (attn / GDN / MLP 56-63 / endpoints)
  **FP8 E4M3 row-scale** (`mma.kind::f8f6f4`, m16n8k32). No software dequant.

## Measured hardware facts (our own probes on this card, not documentation)

MMA tiers, TFLOPS/TOPS, acceptance criterion "32 MMA in the loop body" in SASS:

| tier | peak | x bf16 |
|---|---:|---:|
| bf16 f32acc (today's default) | 255.3 | 1.00 |
| f16 f16acc | 506.6 | 1.98 |
| fp8 e4m3 f32acc | 517.4 | 2.03 |
| fp8 e4m3 f16acc | 1028.9 | 4.03 |
| int8 s32acc | 1026.7 | 4.02 |
| nvfp4 block_scale | 2021.8 | 7.92 |
| 2:4 int8 / 2:4 fp8-f16acc | 1978.3 / 2004.6 | 7.75 / 7.85 |
| 2:4 nvfp4 (m16n8k128) | 3986.2 | 15.61 |

HBM vendor peak 1792 GB/s. L2 = 96 MB (large enough that operator sweeps at T=1024 are
L2-resident — do not mistake >100%-of-HBM numbers for a roofline result). 170 SMs,
65536 registers/SM, 8-register granule, 1536-thread occupancy cap, 99 KiB shared/CTA usable.

## What sm_120a does NOT have (156 ISA probes, ptxas exit 0 + opcode in SASS)

- `wgmma.mma_async` — **absent**. Every Hopper warp-group GEMM scheme is inapplicable verbatim.
- `tcgen05.*` entirely (tensor memory, single-thread MMA, `cta_group`). Papers titled
  "Blackwell GEMM" are usually about `sm_100a`, not this card.
- `cvt.rs` — no hardware stochastic rounding.
- **Block-scaled MMA (`mxf4nvf4` / `mxf4` / `mxf8f6f4`) accepts only an f32 accumulator.**
  "Half accumulator doubles the tier" does NOT work on the fp4 tier. Verified both forms.
- `ldmatrix.m16n16` only `.x1/.x2`; 256-bit access only to global; `applypriority` knows only
  `.evict_normal`; `cvt.rn.bf16x2.e2m1x2` does not exist (fp4 unpacks only to f16).
- Present but unused by NInfer: block clusters + distributed shared memory (`barrier.cluster`,
  `mapa`, `st.async`, `red.async`), `tensormap.replace`, `cp.reduce.async.bulk`, `stmatrix`,
  `ldmatrix.m16n16.trans.b8`, `ldmatrix .b8x16.b4x16_p64`, `redux.sync`, `match.any.sync`,
  `elect.sync`, `discard.global.L2`, `createpolicy`.

## Already measured and REFUTED here — do not propose again without new evidence

- `multicast::cluster` on TMA: **600x slower instruction encoding**, not a bandwidth issue.
  Single-CTA cluster, mask=1, same bytes: `cp.async.bulk` 0.148 ms vs `.multicast` 88.6 ms.
- Deep async pipelines (+9..15% regression), warp specialisation, TMA for paged KV,
  tensor cores in decode attention (118.4 vs 119.0 us), PDL without a device-side half,
  cuTile on sm_120 (3.2% of roofline). (Measured by the neighbouring engine `imp`, reproduced.)
- **ThriftAttention / dynamic sparse attention: closed.** fp4 blockscale (2021.8) vs f16-f16acc
  (506.6)... note the tier table above; the specific closing argument was head_dim 256
  (their promotion path exists only for 64/128) and "our cheapest tier is already exact".
- int4 KV cache: no `s4` tensor core on sm_120; scalar int4 is slower than fp16 everywhere.
- PV via per-tile V conversion to f16 ("route B"): prefill 15086 -> 9565 (**-37%**), because
  the V tile at head_dim 256 is converted by *every* CTA (2048 CTAs per tile at chunk 8192).
- Adaptive draft window: closed, +1.4% on AIME only.
- Tree speculation on MoE: dies. Round cost model 13.78 + 0.180 x columns + 0.590 x steps —
  an extra column is far cheaper than an extra step, so a tree does not pay.

## What NInfer ALREADY has — do not report as a finding

Multi-block sampler; parallel split-K partial reduction; CTA-per-KV-head decode attention;
verification chunking on the decode split-K path (`small_t`, special cases for T=5/6);
`gdn_replay_record`/`gdn_replay_fold` (GDN state rollback on rejection — strictly better than
re-running the forward pass); on-device accept/reject decision (`mtp_prepare_next_round`);
XOR swizzle `Swizzle<3,3,3>` on attention shared memory (no bank conflicts);
Hadamard transform on K and Q (`src/ops/kv_cache/hadamard_d256.cuh:46-65`);
int8 KV cache (K and V quantised **before** they are read, uniformly, on both prefill and decode);
`NINFER_MOE_BN=128` MoE tiling; direct out-projection; W8 shared-scale hoist; W8 row-split;
NVFP4 fused TMA SwiGLU; MoE staging from x; RMSNorm weight hoist above the reduction;
prefill chunking with the good width being 8192 (default 1024 costs 25-30% TTFT).
MTP: draft window up to 15, mtp3 is the canonical fast point (~180 tok/s on 27B,
~579 tok/s on 35B-A3B decode).

## The house rules for a finding

1. **A mechanism is measured before it is implemented.** Report what would be measured and on
   what stand, not "this should be faster".
2. **Cite source.** Every claim about the third-party engine needs `path/file.py:line` from the
   local clone. No claim from memory, no claim from a blog post, no claim from a README that
   the code contradicts.
3. **Say what NInfer does today** for the same thing, with `ninfer/...:line`. If you did not
   check, say "not checked" — do not guess.
4. **Roofline or nothing** for perf claims: give the share of HBM peak (1792 GB/s) or of the
   relevant MMA tier, not just a speedup ratio.
5. Negative results are results. "This is what they do and it cannot help us because X" is a
   valid and valuable finding.

## CORRECTIONS to earlier versions of this brief (verified in the clone, 2026-09-01)

- **NInfer does NOT use f16 accumulation anywhere.** `mma_f16` is
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (`src/ops/common/mma.cuh:42-49`):
  f16 operands, **f32 accumulator**. That is the ~257 TFLOPS tier, not the 506.6 one.
  The PV product in both causal-cache prefill kernels (`prompt_i8.cuh:440`,
  `prompt_fp8.cuh:422`, `small_t_fp8.cuh:539`) already feeds f16 operands from a shared-memory
  `v_f16` tile into an f32 accumulator. So the half-accumulator lever on the PV product is
  **open, not taken** — treat any earlier statement that it is shipped as false.
- **NInfer has a Hadamard transform** on K and Q: `src/ops/kv_cache/hadamard_d256.cuh`.
- `getenv` appears nowhere in `src/`, `include/`, `apps/` — only in `tests/`. Any
  `NINFER_*` environment variable mentioned in campaign notes belongs to a private branch,
  not to upstream. Prefill chunk width is `EngineOptions::prefill_chunk` / `--prefill-chunk`,
  default **1024** (`include/ninfer/types.h:116`), multiple of 128.
- **Prefill is fully eager and drains the pipe after every chunk**:
  `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1312` calls `ctx_.synchronize()`
  (= `cudaStreamSynchronize`, `src/core/device.cu:126`). CUDA graphs are captured for decode
  only (`src/core/decode_graph.cpp:63,111,144`).
