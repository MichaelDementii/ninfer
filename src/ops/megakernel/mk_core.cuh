#pragma once

// NInfer megakernel v0.1 core.
//
// Execution model (design: /root/notes/megakernel_design.md):
//   - One persistent 512-thread CTA per SM; each CTA walks its own pre-built
//     instruction tape (static schedule, reused across steps like a CUDA graph).
//   - Cross-instruction dependencies synchronize through u32 counters in global
//     memory: an instruction's completion posts +1 (release); a consumer spins on
//     lane 0 until the counter reaches its target (acquire), then the CTA joins
//     via __syncthreads().
//   - v0.1 keeps shared memory static per-instruction (no paging yet); paging and
//     cross-instruction weight prefetch arrive in v0.2.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::mk {

inline constexpr int kMkThreads           = 512;
inline constexpr int kMkMaxPtrs           = 8;
inline constexpr int kMkMaxDims           = 8;
inline constexpr int kMkMaxWaits          = 4;
inline constexpr std::uint32_t kMkNone    = 0xffffffffu;
inline constexpr int kMkMaxInstrPerStream = 4096;

enum class MkOp : std::uint32_t {
    Halt = 0,
    RmsNorm2048,     // ptr0=x, ptr1=weight, out0=dst; dim0=rows (each row = 2048 bf16)
    W8GemvResidual,  // ptr0=x, ptr1=codes, ptr2=scales, out0=residual_dst;
                     // dim0=row0, dim1=rows (multiple of warps), dim2=k
    W8DecodeK,       // verbatim w8_k2048_decode body, one warp per output row.
                     // ptr0=x, ptr1=codes, ptr2=scales, out0=dst (plain store) or
                     // residual accumulate when dim3!=0; dim0=row0, dim1=rows,
                     // dim2=K (2048|4096), dim3=residual flag
    Bf16Gemv,        // ptr0=x, ptr1=W(bf16 row-major), out0=dst; dim0=row0,
                     // dim1=rows, dim2=k — small control projections (gating)
    GatedNorm128,    // ptr0=x(base of d=128 rows), ptr1=weight(128), ptr2=z(base),
                     // out0=dst; dim0=row0, dim1=rows, dim2=eps bits — RMSNorm(x)*silu(z)
    SigmoidMul,      // ptr0=v, ptr1=gate(bf16, one per 64 elems), out0=dst;
                     // dim0=elem0, dim1=count — dst = v * sigmoid(gate[i>>6])
    W8DecodeConv,    // w8_k2048_decode body + GdnConvEpilogue (T=1): rows<8192 run the
                     // 4-tap causal conv + silu and split to q/k/v with in-place state
                     // update; rows>=8192 store plain bf16 to z.
                     // ptr0=x, ptr1=codes, ptr2=scales, ptr3=conv_w(4x8192 plane-wise),
                     // ptr4=conv_state(3x8192, updated in place), ptr5=vc, ptr6=z,
                     // out0=qc, out1=kc; dim0=row0, dim1=rows
    GdnRecurrent,    // gated delta net T=1, verbatim per-warp math of
                     // recurrent_bf16_direct_kernel<NormalizeQK=true>. One warp = one
                     // (value_head, 4-row dv tile) unit; unit u: head=u>>5, dv=(u&31)*4.
                     // ptr0=q(16 heads x128), ptr1=k, ptr2=v(32x128), ptr3=g_beta(bf16
                     // 64: g[32]+beta[32]); out0=o(32x128), out1=state(f32 32x128x128);
                     // dim0=unit0, dim1=units, dim2=scale bits
    GdnGating,       // gating transform: ptr0=a(f32 32), ptr1=b(f32 32), ptr2=A_log,
                     // ptr3=dt_bias; out0=g(f32 32), out1=beta(f32 32); dim0=heads
    W8SimtOutProj,   // author's T=1 out-proj (w8_rowsplit_gemm_simt): warp-per-row,
                     // cp.async double-buffered 1024-value slabs, residual epilogue.
                     // ptr0=x(4096), ptr1=codes, ptr2=scales, out0=residual dst;
                     // dim0=row0, dim1=rows
    MoeD1,           // router scores: ptr0=x(2048), ptr1=router(bf16 257x2048),
                     // out0=scores(f32 257); dim0=row0, dim1=rows(2/slice) — two 8-warp
                     // groups per CTA, engine block-reduce order preserved per row
    MoeD2,           // top-8 select + softmax alpha + shared sigmoid scale (1 warp):
                     // ptr0=scores, out0=ids(i32 8), out1=alpha(f32 8); ptr1=shared_scale
    MoeD3,           // gate_up + silu*up: ptr0=x, ptr1=ids, ptr2..4=routed Q4
                     // codes/high/scales, ptr5..6=shared W8 codes/scales,
                     // out0=act(f32 9x512); dim0=j0, dim1=j_count
    MoeD4,           // down + rank-ordered FP32 sum + residual: ptr0=ids, ptr1=alpha,
                     // ptr2=shared_scale, ptr3=act, ptr4..6=routed Q5 codes/high/scales,
                     // ptr7=shared W8 codes, out1=shared W8 scales, out0=destination(x);
                     // dim0=row0, dim1=rows
    FusedGateA,      // author's fused norm+gating mma, one slice = one (split, head
                     // tile) CTA at t=1: ptr0=x, ptr1=norm_w, ptr2=a_w, ptr3=b_w,
                     // out0=partial (32x64 a/b + 32 norm sums); dim0=lin slice
    FusedGateB,      // split reduce in author's order: ptr0=x, ptr1=norm_w,
                     // ptr2=partial, ptr3=A_log, ptr4=dt_bias, ptr5=beta out,
                     // out0=h (bf16 2048), out1=g; dim1=eps bits
    AttnQkv,         // w8_k2048 row dot + W8SplitOutput4<4096,512,4096,512> store:
                     // ptr0=h, ptr1=codes, ptr2=scales, out0=q, out1=k,
                     // ptr5=gate, ptr6=v; dim0=row0, dim1=rows
    NormQK,          // engine rmsnorm_warp<Offset,512> at d=256: warp/row, 4 pairs
                     // per lane, pairwise += sums. ptr0=x, ptr1=w(256), out0=dst;
                     // dim0=rows, dim1=eps bits
    RopeQK,          // rope_fixed_kernel<Text1D,16,2> at T=1: ptr0=positions,
                     // out0=qn (16 heads x 256), out1=kn (2 heads x 256)
    SigGateMul,      // x[i] *= sigmoid(gate[i]) over bf16x8 packs: ptr0=gate,
                     // out0=x; dim0=elements
    Noop,
};

// A tape entry describes a TASK CLASS: `slice_count` independent slices popped
// dynamically by all CTAs through one atomic counter (`task_counter`) — the
// megakernel's answer to the hardware grid scheduler's load balancing (a static
// per-SM assignment measured +17.7% slower on 35B shapes: avg 5.28 slices/SM
// rounds up to 6 on the critical path). Every completed slice posts +1 to
// `done_counter`; slices with index < done2_limit additionally post +1 to
// `done2_counter`, giving consumers a chunk-granular dependency (e.g. out-proj
// only needs the first 256 in-proj slices).
struct MkInstr {
    MkOp op;
    std::uint32_t task_counter;                // atomic pop index for this class
    std::uint32_t slice_count;
    std::uint32_t done_counter;                // kMkNone = do not post
    std::uint32_t done2_counter;               // kMkNone = unused
    std::uint32_t done2_limit;                 // slices with idx < limit post done2
    std::uint32_t wait_counter[kMkMaxWaits];   // kMkNone = slot unused
    std::uint32_t wait_target[kMkMaxWaits];
    const void* ptr[kMkMaxPtrs];
    void* out[2];
    std::int64_t dim[kMkMaxDims];
};

struct MkStream {
    const MkInstr* tape;
    std::uint32_t count;
};

// ---- synchronization ------------------------------------------------------

__device__ __forceinline__ void mk_wait_phase(const MkInstr& instr,
                                              const std::uint32_t* counters) {
    if (threadIdx.x == 0) {
#pragma unroll
        for (int w = 0; w < kMkMaxWaits; ++w) {
            const std::uint32_t idx = instr.wait_counter[w];
            if (idx == kMkNone) { continue; }
            const std::uint32_t target = instr.wait_target[w];
            const volatile std::uint32_t* cell =
                reinterpret_cast<const volatile std::uint32_t*>(&counters[idx]);
            while (*cell < target) { __nanosleep(128); }
        }
    }
    __syncthreads();       // all threads join behind the satisfied waits
    __threadfence();       // acquire: make producers' stores visible to this CTA
}

__device__ __forceinline__ void mk_post_phase(const MkInstr& instr, std::uint32_t* counters) {
    __syncthreads();       // all threads finished their stores
    if (threadIdx.x == 0 && instr.done_counter != kMkNone) {
        __threadfence();   // release: publish stores before the increment lands
        atomicAdd(&counters[instr.done_counter], 1u);
    }
}

// ---- shared storage (v0.1: static union) ----------------------------------

union MkShared {
    struct {
        float warp_sums[kMkThreads / 32];
        float inv;
    } rms;
    struct {
        float partial[2][8];       // two 8-warp router-row groups
    } d1;
    struct {
        float selected_logits[8];  // top-8 selector scratch (single warp)
    } d2;
    struct {
        alignas(16) __nv_bfloat16 x[2048];   // d3 activation staging (engine x_shared)
    } d3;
    struct {
        float paths[9][32];        // rank-ordered FP32 down epilogue, 32 rows/slice
    } d4;
    struct {
        // fused gating stages: 2 x (64x64 x tile + 2 x 16x64 weight tiles) bf16
        alignas(16) __nv_bfloat16 stage[2 * (64 * 64 + 2 * 16 * 64)];
    } fg;
    struct {
        float cs[64];   // rope cos[32] + sin[32] per-token cache
    } rope;
    MkInstr instr_broadcast;
};

// CTA-wide slice pop: thread 0 takes the next index, everyone joins.
__device__ __forceinline__ std::uint32_t mk_pop_slice(std::uint32_t* counters,
                                                      std::uint32_t task_counter,
                                                      std::uint32_t* slice_shared) {
    __syncthreads();
    if (threadIdx.x == 0) { *slice_shared = atomicAdd(&counters[task_counter], 1u); }
    __syncthreads();
    return *slice_shared;
}

} // namespace ninfer::ops::mk
