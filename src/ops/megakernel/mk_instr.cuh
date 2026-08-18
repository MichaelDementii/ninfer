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

// Verbatim transplant of w8_k2048_decode_kernel (ops/linear/w8/w8_k2048_decode.cuh)
// generalized over K: one warp owns one output row, 8 bf16/lane per phase, u16 group
// scales broadcast via shfl(lane>>2). Identical per-row FP order -> bit-exact against
// the engine's decode in-proj path. 16 warps of the interpreter CTA cover
// dim1 (=16) consecutive rows starting at dim0.
template <int K>
__device__ inline void mk_body_w8_decode(const MkInstr& instr) {
    const auto* x      = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* codes  = static_cast<const std::uint8_t*>(instr.ptr[1]);
    const auto* scales = static_cast<const std::uint8_t*>(instr.ptr[2]);
    auto* out          = static_cast<__nv_bfloat16*>(instr.out[0]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];
    const bool residual     = instr.dim[3] != 0;

    constexpr int kGroup             = 32;
    constexpr int kGroupsPerRow      = K / kGroup;
    constexpr int kValuesPerLane     = 8;
    constexpr int kValuesPerPhase    = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase    = kValuesPerPhase / kGroup;
    constexpr int kPhases            = K / kValuesPerPhase;
    constexpr unsigned kFullWarpMask = 0xffffffffu;

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = kMkThreads / 32;

    for (std::int64_t r = warp; r < rows; r += warps) {
        const std::int64_t row        = row0 + r;
        const std::uint8_t* code_row  = codes + row * K;
        const std::uint8_t* scale_row = scales + row * kGroupsPerRow * 2;

        float accumulator = 0.0f;
#pragma unroll
        for (int phase = 0; phase < kPhases; ++phase) {
            unsigned scale_bits = 0;
            if (lane < kGroupsPerPhase) {
                scale_bits = *reinterpret_cast<const std::uint16_t*>(
                    scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
            }
            scale_bits        = __shfl_sync(kFullWarpMask, scale_bits, lane >> 2);
            const float scale = __half2float(__ushort_as_half(scale_bits));

            const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
            const uint2 packed = *reinterpret_cast<const uint2*>(code_row + phase_k);
            float weights[kValuesPerLane];
#pragma unroll
            for (int word_index = 0; word_index < 2; ++word_index) {
                const std::uint32_t word = (&packed.x)[word_index];
                weights[word_index * 4 + 0] =
                    static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
                weights[word_index * 4 + 1] =
                    static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
                weights[word_index * 4 + 2] =
                    static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
                weights[word_index * 4 + 3] =
                    static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
            }

            const uint4 values = *reinterpret_cast<const uint4*>(x + phase_k);
            const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.x));
            const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.y));
            const float2 x2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.z));
            const float2 x3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.w));
            accumulator = fmaf(weights[0], x0.x, accumulator);
            accumulator = fmaf(weights[1], x0.y, accumulator);
            accumulator = fmaf(weights[2], x1.x, accumulator);
            accumulator = fmaf(weights[3], x1.y, accumulator);
            accumulator = fmaf(weights[4], x2.x, accumulator);
            accumulator = fmaf(weights[5], x2.y, accumulator);
            accumulator = fmaf(weights[6], x3.x, accumulator);
            accumulator = fmaf(weights[7], x3.y, accumulator);
        }

        accumulator = mk_warp_reduce_sum(accumulator);
        if (lane == 0) {
            if (residual) {
                const float prior = __bfloat162float(out[row]);
                out[row]          = __float2bfloat16_rn(prior + accumulator);
            } else {
                out[row] = __float2bfloat16_rn(accumulator);
            }
        }
    }
}

// Small control projection: bf16 W[rows x k] @ x, one warp per output row.
__device__ inline void mk_body_bf16_gemv(const MkInstr& instr) {
    const auto* x = static_cast<const __nv_bfloat162*>(instr.ptr[0]);
    const auto* w = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    auto* out     = static_cast<__nv_bfloat16*>(instr.out[0]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];
    const std::int64_t k    = instr.dim[2];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = kMkThreads / 32;

    for (std::int64_t r = warp; r < rows; r += warps) {
        const std::int64_t row = row0 + r;
        const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(w + row * k);
        float acc = 0.0f;
        for (std::int64_t i = lane; i < k / 2; i += 32) {
            const float2 wf = __bfloat1622float2(w2[i]);
            const float2 xf = __bfloat1622float2(x[i]);
            acc = fmaf(wf.x, xf.x, acc);
            acc = fmaf(wf.y, xf.y, acc);
        }
        acc = mk_warp_reduce_sum(acc);
        if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
    }
}

// Gated RMSNorm over d=128 rows (engine: gated_rmsnorm on GDN head outputs):
// one warp per row, 4 bf16x2 pairs per lane; dst = norm(x)*w*silu(z).
__device__ inline void mk_body_gated_norm128(const MkInstr& instr) {
    const auto* x      = static_cast<const __nv_bfloat162*>(instr.ptr[0]);
    const auto* weight = static_cast<const __nv_bfloat162*>(instr.ptr[1]);
    const auto* z      = static_cast<const __nv_bfloat162*>(instr.ptr[2]);
    auto* out          = static_cast<__nv_bfloat162*>(instr.out[0]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];
    const float eps         = __int_as_float(static_cast<int>(instr.dim[2]));

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = kMkThreads / 32;

    for (std::int64_t r = warp; r < rows; r += warps) {
        const std::int64_t base = (row0 + r) * 64;   // 64 bf16x2 pairs per d=128 row
        float2 xf[2];
        const __nv_bfloat162 v0 = x[base + lane];
        const __nv_bfloat162 v1 = x[base + lane + 32];
        xf[0] = __bfloat1622float2(v0);
        xf[1] = __bfloat1622float2(v1);
        float sum = xf[0].x * xf[0].x + xf[0].y * xf[0].y + xf[1].x * xf[1].x +
                    xf[1].y * xf[1].y;
        sum             = mk_warp_reduce_sum(sum);
        float inv       = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
        inv             = __shfl_sync(0xffffffffu, inv, 0);
        const float2 w0 = __bfloat1622float2(weight[lane]);
        const float2 w1 = __bfloat1622float2(weight[lane + 32]);
        const float2 z0 = __bfloat1622float2(z[base + lane]);
        const float2 z1 = __bfloat1622float2(z[base + lane + 32]);
        const auto sil  = [](float v) { return v / (1.0f + __expf(-v)); };
        out[base + lane] = __floats2bfloat162_rn(xf[0].x * inv * (w0.x + 1.0f) * sil(z0.x),
                                                 xf[0].y * inv * (w0.y + 1.0f) * sil(z0.y));
        out[base + lane + 32] =
            __floats2bfloat162_rn(xf[1].x * inv * (w1.x + 1.0f) * sil(z1.x),
                                  xf[1].y * inv * (w1.y + 1.0f) * sil(z1.y));
    }
}

__device__ __forceinline__ float mk_warp_xor_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    }
    return x;
}

// w8_k2048_decode phase loop + GdnConvEpilogue (T=1, in-place conv state update):
// verbatim engine math per row. Rows [0,8192): conv+silu -> q/k/v split, state
// (s0,s1,s2) <- (s1,s2,p). Rows [8192,12288): plain bf16 store to z.
__device__ inline void mk_body_w8_decode_conv(const MkInstr& instr) {
    const auto* x          = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* codes      = static_cast<const std::uint8_t*>(instr.ptr[1]);
    const auto* scales     = static_cast<const std::uint8_t*>(instr.ptr[2]);
    const auto* conv_w     = static_cast<const __nv_bfloat16*>(instr.ptr[3]);
    auto* conv_state       = const_cast<__nv_bfloat16*>(
        static_cast<const __nv_bfloat16*>(instr.ptr[4]));
    auto* vc               = const_cast<__nv_bfloat16*>(
        static_cast<const __nv_bfloat16*>(instr.ptr[5]));
    auto* z                = const_cast<__nv_bfloat16*>(
        static_cast<const __nv_bfloat16*>(instr.ptr[6]));
    auto* qc               = static_cast<__nv_bfloat16*>(instr.out[0]);
    auto* kc               = static_cast<__nv_bfloat16*>(instr.out[1]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];

    constexpr int K                  = 2048;
    constexpr int kChannels          = 8192;
    constexpr int kGroupsPerRow      = K / 32;
    constexpr int kValuesPerLane     = 8;
    constexpr int kValuesPerPhase    = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase    = kValuesPerPhase / 32;
    constexpr int kPhases            = K / kValuesPerPhase;
    constexpr unsigned kFullWarpMask = 0xffffffffu;

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = kMkThreads / 32;

    for (std::int64_t r = warp; r < rows; r += warps) {
        const std::int64_t row        = row0 + r;
        const std::uint8_t* code_row  = codes + row * K;
        const std::uint8_t* scale_row = scales + row * kGroupsPerRow * 2;

        float accumulator = 0.0f;
#pragma unroll
        for (int phase = 0; phase < kPhases; ++phase) {
            unsigned scale_bits = 0;
            if (lane < kGroupsPerPhase) {
                scale_bits = *reinterpret_cast<const std::uint16_t*>(
                    scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
            }
            scale_bits        = __shfl_sync(kFullWarpMask, scale_bits, lane >> 2);
            const float scale = __half2float(__ushort_as_half(scale_bits));
            const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
            const uint2 packed = *reinterpret_cast<const uint2*>(code_row + phase_k);
            float weights[kValuesPerLane];
#pragma unroll
            for (int word_index = 0; word_index < 2; ++word_index) {
                const std::uint32_t word = (&packed.x)[word_index];
                weights[word_index * 4 + 0] =
                    static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
                weights[word_index * 4 + 1] =
                    static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
                weights[word_index * 4 + 2] =
                    static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
                weights[word_index * 4 + 3] =
                    static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
            }
            const uint4 values = *reinterpret_cast<const uint4*>(x + phase_k);
            const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.x));
            const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.y));
            const float2 x2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.z));
            const float2 x3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&values.w));
            accumulator = fmaf(weights[0], x0.x, accumulator);
            accumulator = fmaf(weights[1], x0.y, accumulator);
            accumulator = fmaf(weights[2], x1.x, accumulator);
            accumulator = fmaf(weights[3], x1.y, accumulator);
            accumulator = fmaf(weights[4], x2.x, accumulator);
            accumulator = fmaf(weights[5], x2.y, accumulator);
            accumulator = fmaf(weights[6], x3.x, accumulator);
            accumulator = fmaf(weights[7], x3.y, accumulator);
        }
        accumulator = mk_warp_reduce_sum(accumulator);

        if (lane == 0) {
            if (row < kChannels) {
                const float s0 = __bfloat162float(conv_state[row]);
                const float s1 = __bfloat162float(conv_state[kChannels + row]);
                const float s2 = __bfloat162float(conv_state[2 * kChannels + row]);
                const float w0 = __bfloat162float(conv_w[row]);
                const float w1 = __bfloat162float(conv_w[kChannels + row]);
                const float w2 = __bfloat162float(conv_w[2 * kChannels + row]);
                const float w3 = __bfloat162float(conv_w[3 * kChannels + row]);
                float conv     = fmaf(w0, s0, 0.0f);
                conv           = fmaf(w1, s1, conv);
                conv           = fmaf(w2, s2, conv);
                conv           = fmaf(w3, accumulator, conv);
                const float sil            = conv / (1.0f + __expf(-conv));
                const __nv_bfloat16 output = __float2bfloat16_rn(sil);
                if (row < 2048) {
                    qc[row] = output;
                } else if (row < 4096) {
                    kc[row - 2048] = output;
                } else {
                    vc[row - 4096] = output;
                }
                conv_state[row]                 = __float2bfloat16_rn(s1);
                conv_state[kChannels + row]     = __float2bfloat16_rn(s2);
                conv_state[2 * kChannels + row] = __float2bfloat16_rn(accumulator);
            } else {
                z[row - kChannels] = __float2bfloat16_rn(accumulator);
            }
        }
    }
}

// Gated delta net T=1: verbatim per-warp math of recurrent_bf16_direct_kernel
// <NormalizeQK=true> (state tile in registers, xor-butterfly partials, shfl_down
// L2 normalization) — the engine kernel has NO cross-warp traffic, so one warp
// here = one (value_head, 4-row dv tile) unit, bit-exact per unit.
__device__ inline void mk_body_gdn_recurrent(const MkInstr& instr) {
    const auto* q   = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* k   = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    const auto* v   = static_cast<const __nv_bfloat16*>(instr.ptr[2]);
    const auto* gb  = static_cast<const __nv_bfloat16*>(instr.ptr[3]);
    auto* out       = static_cast<__nv_bfloat16*>(instr.out[0]);
    auto* state     = static_cast<float*>(instr.out[1]);
    const std::int64_t unit0 = instr.dim[0];
    const std::int64_t units = instr.dim[1];
    const float scale        = __int_as_float(static_cast<int>(instr.dim[2]));

    constexpr int kStateDim = 128;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int warps         = kMkThreads / 32;
    const int dqk_base      = lane * 4;

    for (std::int64_t u = unit0 + warp; u < unit0 + units; u += warps) {
        const int head    = static_cast<int>(u >> 5);
        const int dv_base = static_cast<int>(u & 31) * 4;
        const int h_qk    = head >> 1;   // 32 value heads share 16 qk heads

        float* state_h = state + static_cast<std::int64_t>(head) * kStateDim * kStateDim;
        float st[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4 sv = *reinterpret_cast<const float4*>(
                state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim + dqk_base);
            st[r][0] = sv.x;
            st[r][1] = sv.y;
            st[r][2] = sv.z;
            st[r][3] = sv.w;
        }

        const auto load_qk_norm = [&](const __nv_bfloat16* base, float (&reg)[4]) {
            const __nv_bfloat162 p0 =
                *reinterpret_cast<const __nv_bfloat162*>(base + dqk_base);
            const __nv_bfloat162 p1 =
                *reinterpret_cast<const __nv_bfloat162*>(base + dqk_base + 2);
            const float2 lo = __bfloat1622float2(p0);
            const float2 hi = __bfloat1622float2(p1);
            reg[0]          = lo.x;
            reg[1]          = lo.y;
            reg[2]          = hi.x;
            reg[3]          = hi.y;
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { sum += reg[i] * reg[i]; }
            sum       = mk_warp_reduce_sum(sum);
            float inv = lane == 0 ? rsqrtf(sum + 1.0e-6f) : 0.0f;
            inv       = __shfl_sync(0xffffffffu, inv, 0);
#pragma unroll
            for (int i = 0; i < 4; ++i) { reg[i] *= inv; }
        };

        float key[4];
        load_qk_norm(k + static_cast<std::int64_t>(h_qk) * kStateDim, key);

        const float g_val    = __bfloat162float(gb[head]);
        const float beta_val = __bfloat162float(gb[32 + head]);
        const float alpha    = expf(g_val);

        float v_local = 0.0f;
        if (lane < 4) {
            v_local = __bfloat162float(v[static_cast<std::int64_t>(head) * kStateDim +
                                         dv_base + lane]);
        }

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) { partial += st[r][c] * key[c]; }
            partial           = mk_warp_xor_sum(partial);
            const float v_r   = __shfl_sync(0xffffffffu, v_local, r);
            const float delta = beta_val * (v_r - alpha * partial);
#pragma unroll
            for (int c = 0; c < 4; ++c) { st[r][c] = alpha * st[r][c] + delta * key[c]; }
        }

        float query[4];
        load_qk_norm(q + static_cast<std::int64_t>(h_qk) * kStateDim, query);
        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) { partial += st[r][c] * query[c]; }
            partial = mk_warp_xor_sum(partial);
            if (lane == r) { attn_val = partial; }
        }
        if (lane < 4) {
            out[static_cast<std::int64_t>(head) * kStateDim + dv_base + lane] =
                __float2bfloat16(attn_val * scale);
        }

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            *reinterpret_cast<float4*>(state_h +
                                       static_cast<std::int64_t>(dv_base + r) * kStateDim +
                                       dqk_base) =
                make_float4(st[r][0], st[r][1], st[r][2], st[r][3]);
        }
    }
}

// dst[i] = v[i] * sigmoid(gate[i >> 6]) over dim1 elements.
__device__ inline void mk_body_sigmoid_mul(const MkInstr& instr) {
    const auto* v    = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* gate = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    auto* out        = static_cast<__nv_bfloat16*>(instr.out[0]);
    const std::int64_t e0    = instr.dim[0];
    const std::int64_t count = instr.dim[1];
    for (std::int64_t i = e0 + threadIdx.x; i < e0 + count; i += kMkThreads) {
        const float g = __bfloat162float(gate[i >> 6]);
        const float s = 1.0f / (1.0f + __expf(-g));
        out[i]        = __float2bfloat16_rn(__bfloat162float(v[i]) * s);
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
    case MkOp::W8DecodeK:
        if (instr.dim[2] == 2048) {
            mk_body_w8_decode<2048>(instr);
        } else {
            mk_body_w8_decode<4096>(instr);
        }
        return;
    case MkOp::Bf16Gemv:
        mk_body_bf16_gemv(instr);
        return;
    case MkOp::GatedNorm128:
        mk_body_gated_norm128(instr);
        return;
    case MkOp::SigmoidMul:
        mk_body_sigmoid_mul(instr);
        return;
    case MkOp::W8DecodeConv:
        mk_body_w8_decode_conv(instr);
        return;
    case MkOp::GdnRecurrent:
        mk_body_gdn_recurrent(instr);
        return;
    case MkOp::Halt:
    case MkOp::Noop:
        return;
    }
}

__global__ __launch_bounds__(kMkThreads, 1) void mk_ref_generic_kernel(MkInstr instr) {
    __shared__ MkShared shared;
    MkInstr local = instr;
    local.dim[0]  = instr.dim[0] + static_cast<std::int64_t>(blockIdx.x) * instr.dim[1];
    mk_execute(local, shared);
}

// ---- the interpreter -------------------------------------------------------

// While lane 0 spins on dependency counters, the other threads pull one slice's
// worth of this class's WEIGHT bytes into L2 — weights are immutable, so
// prefetching ahead of the data dependency is always safe, and the DRAM fetch
// overlaps the wait instead of following it. With dynamic slice popping the
// exact future assignment is unknown; the CTA's grid rank approximates its
// first-wave slice.
__device__ __forceinline__ void mk_prefetch_slice(const MkInstr& instr, unsigned rank) {
    if (instr.op != MkOp::W8DecodeK) { return; }
    if (rank >= instr.slice_count) { return; }
    const auto* codes        = static_cast<const char*>(instr.ptr[1]);
    const std::int64_t row0  = instr.dim[0] + static_cast<std::int64_t>(rank) * instr.dim[1];
    const std::int64_t base  = row0 * instr.dim[2];
    const std::int64_t bytes = instr.dim[1] * instr.dim[2];
    for (std::int64_t off = static_cast<std::int64_t>(threadIdx.x) * 128; off < bytes;
         off += static_cast<std::int64_t>(kMkThreads) * 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(codes + base + off));
    }
}

__global__ __launch_bounds__(kMkThreads, 1) void mk_interpreter_kernel(
    const MkStream* __restrict__ streams, std::uint32_t* __restrict__ counters,
    int enable_prefetch) {
    __shared__ MkShared shared;
    __shared__ std::uint32_t slice_shared;
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
        if (enable_prefetch != 0) { mk_prefetch_slice(instr_s, blockIdx.x); }
        mk_wait_phase(instr_s, counters);
        // First pop; afterwards the done-post of slice N and the pop of slice N+1
        // share one thread-0 critical section (2 barriers per slice, not 3).
        if (threadIdx.x == 0) { slice_shared = atomicAdd(&counters[instr_s.task_counter], 1u); }
        __syncthreads();
        std::uint32_t idx = slice_shared;
        while (idx < instr_s.slice_count) {
            // No in-loop prefetch: with DRAM already saturated by demand streaming,
            // extra prefetch instructions only add memory-pipe pressure (measured
            // +10% regression). Prefetch pays off solely inside dependency waits,
            // where the bus would otherwise idle.
            MkInstr local = instr_s;
            local.dim[0] = instr_s.dim[0] + static_cast<std::int64_t>(idx) * instr_s.dim[1];
            mk_execute(local, shared);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                if (instr_s.done_counter != kMkNone) {
                    atomicAdd(&counters[instr_s.done_counter], 1u);
                }
                if (instr_s.done2_counter != kMkNone && idx < instr_s.done2_limit) {
                    atomicAdd(&counters[instr_s.done2_counter], 1u);
                }
                slice_shared = atomicAdd(&counters[instr_s.task_counter], 1u);
            }
            __syncthreads();
            idx = slice_shared;
        }
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

__global__ __launch_bounds__(kMkThreads, 1) void mk_ref_w8_decode_kernel(MkInstr instr) {
    MkInstr local = instr;
    local.dim[0]  = instr.dim[0] + static_cast<std::int64_t>(blockIdx.x) * instr.dim[1];
    if (instr.dim[2] == 2048) {
        mk_body_w8_decode<2048>(local);
    } else {
        mk_body_w8_decode<4096>(local);
    }
}

} // namespace ninfer::ops::mk
