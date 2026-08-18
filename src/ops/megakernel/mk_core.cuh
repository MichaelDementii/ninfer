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
    // W8 GEMV keeps everything in registers/L2 in v0.1 (no staging yet).
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
