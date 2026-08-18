#pragma once

// Megakernel v0.1 instruction bodies.
//
// Every body is a plain __device__ function over (instruction, threadIdx) so the
// SAME code runs in two harness modes: inside the persistent interpreter and as a
// standalone per-op kernel (reference). Identical partitioning + identical
// reduction order in both modes -> outputs must match bit-for-bit; the timing
// difference isolates pure kernel-boundary overhead.
//
// RmsNorm2048 is a verbatim transplant of the engine's rmsnorm_d2048_bf16x2_kernel
// (Offset epilogue, one 512-thread CTA per row) — the same body the real decode
// path runs at T=1, so the transplant stays bit-exact against the engine op.

#include "mk_core.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::mk {

__device__ __forceinline__ float mk_warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

__device__ __forceinline__ float mk_block_reduce_sum_512(float x, float* warp_sums) {
    x = mk_warp_reduce_sum(x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { warp_sums[warp] = x; }
    __syncthreads();
    x = threadIdx.x < 16 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            x += __shfl_down_sync(0xffffffffu, x, offset, 16);
        }
    }
    return x;
}

// dim0 = rows; one full CTA per row, rows processed sequentially by this CTA.
// Matches rmsnorm_d2048_bf16x2_kernel<Offset> with rows folded into a loop.
__device__ inline void mk_body_rmsnorm2048(const MkInstr& instr, MkShared& shared) {
    const auto* x       = static_cast<const __nv_bfloat162*>(instr.ptr[0]);
    const auto* weight  = static_cast<const __nv_bfloat162*>(instr.ptr[1]);
    auto* out           = static_cast<__nv_bfloat162*>(instr.out[0]);
    const std::int64_t rows = instr.dim[0];
    const float eps         = __int_as_float(static_cast<int>(instr.dim[1]));

    constexpr int kBlock       = kMkThreads;
    constexpr int kPairsPerRow = 1024;

    for (std::int64_t row = 0; row < rows; ++row) {
        const std::int64_t row_base = row * kPairsPerRow;
        const int pair0             = static_cast<int>(threadIdx.x);
        const int pair1             = pair0 + kBlock;
        const __nv_bfloat162 value0 = x[row_base + pair0];
        const __nv_bfloat162 value1 = x[row_base + pair1];
        const float2 x0             = __bfloat1622float2(value0);
        const float2 x1             = __bfloat1622float2(value1);
        const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

        const float sum = mk_block_reduce_sum_512(local_sum, shared.rms.warp_sums);
        if (threadIdx.x == 0) { shared.rms.inv = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
        __syncthreads();
        const float inv = shared.rms.inv;

        const float2 w0 = __bfloat1622float2(weight[pair0]);
        const float2 w1 = __bfloat1622float2(weight[pair1]);
        out[row_base + pair0] = __floats2bfloat162_rn(x0.x * inv * (w0.x + 1.0f),
                                                      x0.y * inv * (w0.y + 1.0f));
        out[row_base + pair1] = __floats2bfloat162_rn(x1.x * inv * (w1.x + 1.0f),
                                                      x1.y * inv * (w1.y + 1.0f));
        __syncthreads();
    }
}

// W8G32 GEMV + residual accumulate. dim0=row0, dim1=rows (multiple of 16),
// dim2=k. One warp owns one output row; 16 warps -> 16 rows per instruction.
// Weights: s8 codes row-major (k per row), fp16 scale per 32-wide group.
__device__ inline void mk_body_w8_gemv_residual(const MkInstr& instr) {
    const auto* x      = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* codes  = static_cast<const std::int8_t*>(instr.ptr[1]);
    const auto* scales = static_cast<const std::uint16_t*>(instr.ptr[2]);
    auto* out          = static_cast<__nv_bfloat16*>(instr.out[0]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];
    const std::int64_t k    = instr.dim[2];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warps  = kMkThreads / 32;
    const std::int64_t groups = k / 32;

    for (std::int64_t r = warp; r < rows; r += warps) {
        const std::int64_t row = row0 + r;
        const std::int8_t* row_codes    = codes + row * k;
        const std::uint16_t* row_scales = scales + row * groups;
        float acc = 0.0f;
        for (std::int64_t g = 0; g < groups; ++g) {
            const float scale = __half2float(__ushort_as_half(row_scales[g]));
            const float code  = static_cast<float>(row_codes[g * 32 + lane]);
            const float xv    = __bfloat162float(x[g * 32 + lane]);
            acc = fmaf(code * scale, xv, acc);
        }
        acc = mk_warp_reduce_sum(acc);
        if (lane == 0) {
            const float residual = __bfloat162float(out[row]);
            out[row]             = __float2bfloat16_rn(residual + acc);
        }
    }
}

__device__ __forceinline__ void mk_execute(const MkInstr& instr, MkShared& shared) {
    switch (instr.op) {
    case MkOp::RmsNorm2048:
        mk_body_rmsnorm2048(instr, shared);
        return;
    case MkOp::W8GemvResidual:
        mk_body_w8_gemv_residual(instr);
        return;
    case MkOp::Halt:
    case MkOp::Noop:
        return;
    }
}

// ---- the interpreter -------------------------------------------------------

__global__ __launch_bounds__(kMkThreads, 1) void mk_interpreter_kernel(
    const MkStream* __restrict__ streams, std::uint32_t* __restrict__ counters) {
    __shared__ MkShared shared;
    const MkStream stream = streams[blockIdx.x];
    for (std::uint32_t i = 0; i < stream.count; ++i) {
        __shared__ __align__(16) MkInstr instr_s;
        // cooperative broadcast of the descriptor through shared memory
        {
            const auto* src = reinterpret_cast<const std::uint32_t*>(&stream.tape[i]);
            auto* dst       = reinterpret_cast<std::uint32_t*>(&instr_s);
            for (unsigned w = threadIdx.x; w * 4 < sizeof(MkInstr); w += blockDim.x) {
                dst[w] = src[w];
            }
            __syncthreads();
        }
        mk_wait_phase(instr_s, counters);
        mk_execute(instr_s, shared);
        mk_post_phase(instr_s, counters);
    }
}

// ---- standalone per-op reference kernels (same bodies, one launch per op) ---

__global__ __launch_bounds__(kMkThreads, 1) void mk_ref_rmsnorm_kernel(MkInstr instr) {
    __shared__ MkShared shared;
    mk_body_rmsnorm2048(instr, shared);
}

__global__ __launch_bounds__(kMkThreads, 1) void mk_ref_w8_gemv_kernel(MkInstr instr) {
    // grid.x CTAs each take a 16-row slice, mirroring one tape instruction per SM
    MkInstr local = instr;
    local.dim[0]  = instr.dim[0] + static_cast<std::int64_t>(blockIdx.x) * instr.dim[1];
    mk_body_w8_gemv_residual(local);
}

} // namespace ninfer::ops::mk
