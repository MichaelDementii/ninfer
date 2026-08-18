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
    Noop,
};

struct MkInstr {
    MkOp op;
    std::uint32_t done_counter;                // kMkNone = do not post
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

} // namespace ninfer::ops::mk
