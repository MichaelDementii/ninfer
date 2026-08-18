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
#include <math_constants.h>

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
// dim3 != 0 stores FP32 (engine gating feeds the transform in fp32).
__device__ inline void mk_body_bf16_gemv(const MkInstr& instr) {
    const auto* x = static_cast<const __nv_bfloat162*>(instr.ptr[0]);
    const auto* w = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];
    const std::int64_t k    = instr.dim[2];
    const bool f32_out      = instr.dim[3] != 0;

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
        if (lane == 0) {
            if (f32_out) {
                static_cast<float*>(instr.out[0])[row] = acc;
            } else {
                static_cast<__nv_bfloat16*>(instr.out[0])[row] = __float2bfloat16_rn(acc);
            }
        }
    }
}

// GDN gating transform (engine gdn_gating math, 35B: 32 value heads):
// g[h] = -expf(A_log[h]) * softplus(a[h] + dt_bias[h]); beta[h] = sigmoid(b[h]).
__device__ inline void mk_body_gdn_gating(const MkInstr& instr) {
    const auto* a       = static_cast<const float*>(instr.ptr[0]);
    const auto* b       = static_cast<const float*>(instr.ptr[1]);
    const auto* a_log   = static_cast<const float*>(instr.ptr[2]);
    const auto* dt_bias = static_cast<const float*>(instr.ptr[3]);
    auto* g             = static_cast<float*>(instr.out[0]);
    auto* beta          = static_cast<float*>(instr.out[1]);
    const int heads     = static_cast<int>(instr.dim[0]);
    for (int h = static_cast<int>(threadIdx.x); h < heads; h += kMkThreads) {
        const float av = a[h] + dt_bias[h];
        const float sp = av > 20.0f ? av : log1pf(expf(av));
        g[h]           = -expf(a_log[h]) * sp;
        beta[h]        = 1.0f / (1.0f + expf(-b[h]));
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
        const auto sil  = [](float v) { return v / (1.0f + expf(-v)); };
        out[base + lane] = __floats2bfloat162_rn(xf[0].x * inv * w0.x * sil(z0.x),
                                                 xf[0].y * inv * w0.y * sil(z0.y));
        out[base + lane + 32] =
            __floats2bfloat162_rn(xf[1].x * inv * w1.x * sil(z1.x),
                                  xf[1].y * inv * w1.y * sil(z1.y));
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
                const auto* slots =
                    static_cast<const std::int32_t*>(instr.ptr[7]);   // nullptr => slot 0
                const std::int64_t slot_off =
                    slots != nullptr
                        ? static_cast<std::int64_t>(slots[0]) * (3LL * kChannels)
                        : 0;
                __nv_bfloat16* slot_state = conv_state + slot_off;
                const float s0 = __bfloat162float(slot_state[row]);
                const float s1 = __bfloat162float(slot_state[kChannels + row]);
                const float s2 = __bfloat162float(slot_state[2 * kChannels + row]);
                const float w0 = __bfloat162float(conv_w[row]);
                const float w1 = __bfloat162float(conv_w[kChannels + row]);
                const float w2 = __bfloat162float(conv_w[2 * kChannels + row]);
                const float w3 = __bfloat162float(conv_w[3 * kChannels + row]);
                float conv     = fmaf(w0, s0, 0.0f);
                conv           = fmaf(w1, s1, conv);
                conv           = fmaf(w2, s2, conv);
                conv           = fmaf(w3, accumulator, conv);
                const float sil            = conv / (1.0f + expf(-conv));
                const __nv_bfloat16 output = __float2bfloat16_rn(sil);
                if (row < 2048) {
                    qc[row] = output;
                } else if (row < 4096) {
                    kc[row - 2048] = output;
                } else {
                    vc[row - 4096] = output;
                }
                slot_state[row]                 = __float2bfloat16_rn(s1);
                slot_state[kChannels + row]     = __float2bfloat16_rn(s2);
                slot_state[2 * kChannels + row] = __float2bfloat16_rn(accumulator);
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
    const auto* q             = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* k             = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    const auto* v             = static_cast<const __nv_bfloat16*>(instr.ptr[2]);
    const auto* g_arr         = static_cast<const float*>(instr.ptr[3]);
    const auto* beta_arr      = static_cast<const float*>(instr.ptr[4]);
    const auto* initial_slots = static_cast<const std::int32_t*>(instr.ptr[5]);
    const auto* snapshot_base = static_cast<const std::int32_t*>(instr.ptr[6]);
    auto* out                 = static_cast<__nv_bfloat16*>(instr.out[0]);
    auto* state               = static_cast<float*>(instr.out[1]);
    const std::int64_t unit0  = instr.dim[0];
    const std::int64_t units  = instr.dim[1];
    const float scale         = __int_as_float(static_cast<int>(instr.dim[2]));
    const std::int64_t slot_stride = instr.dim[3];   // 0 => single-slot direct (harness)

    constexpr int kStateDim = 128;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int warps         = kMkThreads / 32;
    const int dqk_base      = lane * 4;

    const std::int64_t read_slot_off =
        initial_slots != nullptr ? static_cast<std::int64_t>(initial_slots[0]) * slot_stride : 0;
    const std::int64_t write_slot_off =
        snapshot_base != nullptr ? static_cast<std::int64_t>(snapshot_base[0]) * slot_stride : 0;

    for (std::int64_t u = unit0 + warp; u < unit0 + units; u += warps) {
        const int head    = static_cast<int>(u >> 5);
        const int dv_base = static_cast<int>(u & 31) * 4;
        const int h_qk    = head >> 1;   // 32 value heads share 16 qk heads

        const float* state_read = state + read_slot_off +
                                  static_cast<std::int64_t>(head) * kStateDim * kStateDim;
        float* state_write = state + write_slot_off +
                             static_cast<std::int64_t>(head) * kStateDim * kStateDim;
        float st[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4 sv = *reinterpret_cast<const float4*>(
                state_read + static_cast<std::int64_t>(dv_base + r) * kStateDim + dqk_base);
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

        const float g_val    = g_arr[head];
        const float beta_val = beta_arr[head];
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
            *reinterpret_cast<float4*>(state_write +
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
        const float s = 1.0f / (1.0f + expf(-g));
        out[i]        = __float2bfloat16_rn(__bfloat162float(v[i]) * s);
    }
}

// ==== sparse MoE (T=1), verbatim ports of sparse_moe_decode_kernels.cu =======

namespace moe {

constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = kExperts + 1;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;

__device__ __forceinline__ float2 mk_bf16x2_bits_to_float2(std::uint32_t bits) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
}

__device__ __forceinline__ __half2 mk_half2_from_bits(std::uint32_t bits) {
    __half2 value;
    *reinterpret_cast<std::uint32_t*>(&value) = bits;
    return value;
}

__device__ __forceinline__ float mk_silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float mk_dot_bf16_eight(const __nv_bfloat16* a,
                                                   const __nv_bfloat16* b) {
    const uint4 av  = *reinterpret_cast<const uint4*>(a);
    const uint4 bv  = *reinterpret_cast<const uint4*>(b);
    const float2 a0 = mk_bf16x2_bits_to_float2(av.x);
    const float2 a1 = mk_bf16x2_bits_to_float2(av.y);
    const float2 a2 = mk_bf16x2_bits_to_float2(av.z);
    const float2 a3 = mk_bf16x2_bits_to_float2(av.w);
    const float2 b0 = mk_bf16x2_bits_to_float2(bv.x);
    const float2 b1 = mk_bf16x2_bits_to_float2(bv.y);
    const float2 b2 = mk_bf16x2_bits_to_float2(bv.z);
    const float2 b3 = mk_bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

__device__ __forceinline__ void q4_decode_eight(std::uint32_t packed, std::uint16_t scale_bits,
                                                float (&weights)[8]) {
    const std::uint32_t word = packed ^ 0x88888888u;
    const float scale        = __half2float(__ushort_as_half(scale_bits));
    const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        const __half2 decoded    = __hsub2(mk_half2_from_bits(bits), bias);
        const float2 values      = __half22float2(decoded);
        weights[pair]            = values.x * scale;
        weights[pair + 4]        = values.y * scale;
    }
}

__device__ __forceinline__ void q5_decode_eight(std::uint32_t packed, std::uint8_t high_bits,
                                                std::uint16_t scale_bits, float (&weights)[8]) {
    const std::uint32_t high = static_cast<std::uint32_t>(high_bits) ^ 0xffu;
    const float scale        = __half2float(__ushort_as_half(scale_bits));
    const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        bits |= (((high >> pair) & 1u) << 4) | (((high >> (pair + 4)) & 1u) << 20);
        const __half2 decoded = __hsub2(mk_half2_from_bits(bits), bias);
        const float2 values   = __half22float2(decoded);
        weights[pair]         = values.x * scale;
        weights[pair + 4]     = values.y * scale;
    }
}

__device__ __forceinline__ void q6_decode_eight(std::uint32_t packed, std::uint16_t high_bits,
                                                std::uint16_t scale_bits, float (&weights)[8]) {
    const std::uint32_t high = static_cast<std::uint32_t>(high_bits) ^ 0xaaaau;
    const float scale        = __half2float(__ushort_as_half(scale_bits));
    const __half2 bias       = __half2half2(__ushort_as_half(0x6420)); // 1056.0
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        bits |= (((high >> (2 * pair)) & 3u) << 4) | (((high >> (2 * pair + 8)) & 3u) << 20);
        const __half2 decoded = __hsub2(mk_half2_from_bits(bits), bias);
        const float2 values   = __half22float2(decoded);
        weights[pair]         = values.x * scale;
        weights[pair + 4]     = values.y * scale;
    }
}

// dot_two_rows<Q4Codec, 2048>: 8-value lane ownership, four adjacent groups per
// warp transaction (verbatim engine loop, x read straight from global).
__device__ __forceinline__ void q4_dot_two_rows(const std::uint8_t* codes,
                                                const std::uint8_t* scales, int row0, int row1,
                                                const __nv_bfloat16* x, float& result0,
                                                float& result1) {
    constexpr int kGroups   = kHidden / 64;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int lane_group    = lane >> 3;
    const int lane_in_group = lane & 7;
    float acc0              = 0.0f;
    float acc1              = 0.0f;
    for (int group_base = 0; group_base < kGroups; group_base += 4) {
        const int group           = group_base + lane_group;
        const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
        const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
        const std::uint32_t packed0 =
            *reinterpret_cast<const std::uint32_t*>(codes + index0 * 32 + lane_in_group * 4);
        const std::uint32_t packed1 =
            *reinterpret_cast<const std::uint32_t*>(codes + index1 * 32 + lane_in_group * 4);
        const auto scale0 = *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2);
        const auto scale1 = *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2);
        float weights0[8];
        float weights1[8];
        q4_decode_eight(packed0, scale0, weights0);
        q4_decode_eight(packed1, scale1, weights1);
        const uint4 input     = *reinterpret_cast<const uint4*>(x + group * 64 + lane_in_group * 8);
        const float2 x0       = mk_bf16x2_bits_to_float2(input.x);
        const float2 x1       = mk_bf16x2_bits_to_float2(input.y);
        const float2 x2       = mk_bf16x2_bits_to_float2(input.z);
        const float2 x3       = mk_bf16x2_bits_to_float2(input.w);
        const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            acc0 = fmaf(weights0[item], values[item], acc0);
            acc1 = fmaf(weights1[item], values[item], acc1);
        }
    }
    result0 = mk_warp_reduce_sum(acc0);
    result1 = mk_warp_reduce_sum(acc1);
}

// dot_two_rows<W8Codec, 2048>: one value per lane per 32-group (verbatim).
__device__ __forceinline__ void w8_dot_two_rows(const std::uint8_t* codes,
                                                const std::uint8_t* scales, int row0, int row1,
                                                const __nv_bfloat16* x, float& result0,
                                                float& result1) {
    constexpr int kGroups = kHidden / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    for (int group = 0; group < kGroups; ++group) {
        const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
        const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
        const float s0            = __half2float(__ushort_as_half(
            *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2)));
        const float s1            = __half2float(__ushort_as_half(
            *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2)));
        const float w0 =
            static_cast<float>(static_cast<std::int8_t>(codes[index0 * 32 + lane])) * s0;
        const float w1 =
            static_cast<float>(static_cast<std::int8_t>(codes[index1 * 32 + lane])) * s1;
        const float xv = __bfloat162float(x[group * 32 + lane]);
        acc0           = fmaf(w0, xv, acc0);
        acc1           = fmaf(w1, xv, acc1);
    }
    result0 = mk_warp_reduce_sum(acc0);
    result1 = mk_warp_reduce_sum(acc1);
}

// Two j-columns of one expert share every x load: gate/up for j and j+1 in one
// sweep (4 accumulators). Per-output FP order identical to the single version.
__device__ __forceinline__ void q4_dot_two_rows_x2(const std::uint8_t* codes,
                                                   const std::uint8_t* scales, int rg0, int ru0,
                                                   int rg1, int ru1, const __nv_bfloat16* x,
                                                   float (&result)[4]) {
    constexpr int kGroups   = kHidden / 64;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int lane_group    = lane >> 3;
    const int lane_in_group = lane & 7;
    float acc[4]            = {0.0f, 0.0f, 0.0f, 0.0f};
    const int rows[4]       = {rg0, ru0, rg1, ru1};
    for (int group_base = 0; group_base < kGroups; group_base += 4) {
        const int group   = group_base + lane_group;
        const uint4 input = *reinterpret_cast<const uint4*>(x + group * 64 + lane_in_group * 8);
        const float2 x0   = mk_bf16x2_bits_to_float2(input.x);
        const float2 x1   = mk_bf16x2_bits_to_float2(input.y);
        const float2 x2   = mk_bf16x2_bits_to_float2(input.z);
        const float2 x3   = mk_bf16x2_bits_to_float2(input.w);
        const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const std::int64_t index = static_cast<std::int64_t>(rows[r]) * kGroups + group;
            const std::uint32_t packed =
                *reinterpret_cast<const std::uint32_t*>(codes + index * 32 + lane_in_group * 4);
            const auto scale = *reinterpret_cast<const std::uint16_t*>(scales + index * 2);
            float weights[8];
            q4_decode_eight(packed, scale, weights);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc[r] = fmaf(weights[item], values[item], acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) { result[r] = mk_warp_reduce_sum(acc[r]); }
}

__device__ __forceinline__ void w8_dot_two_rows_x2(const std::uint8_t* codes,
                                                   const std::uint8_t* scales, int rg0, int ru0,
                                                   int rg1, int ru1, const __nv_bfloat16* x,
                                                   float (&result)[4]) {
    constexpr int kGroups = kHidden / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[4]          = {0.0f, 0.0f, 0.0f, 0.0f};
    const int rows[4]     = {rg0, ru0, rg1, ru1};
    for (int group = 0; group < kGroups; ++group) {
        const float xv = __bfloat162float(x[group * 32 + lane]);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const std::int64_t index = static_cast<std::int64_t>(rows[r]) * kGroups + group;
            const float s            = __half2float(__ushort_as_half(
                *reinterpret_cast<const std::uint16_t*>(scales + index * 2)));
            const float w =
                static_cast<float>(static_cast<std::int8_t>(codes[index * 32 + lane])) * s;
            acc[r] = fmaf(w, xv, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) { result[r] = mk_warp_reduce_sum(acc[r]); }
}

// Two down-rows share every activation load.
__device__ __forceinline__ void q5_dot_fp32_row_x2(const std::uint8_t* codes,
                                                   const std::uint8_t* high,
                                                   const std::uint8_t* scales, int row0, int row1,
                                                   const float* x, float& r0, float& r1) {
    constexpr int kGroups   = kIntermediate / 64;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int lane_group    = lane >> 3;
    const int lane_in_group = lane & 7;
    float acc0              = 0.0f;
    float acc1              = 0.0f;
    for (int group_base = 0; group_base < kGroups; group_base += 4) {
        const int group = group_base + lane_group;
        const float4 x0 = *reinterpret_cast<const float4*>(x + group * 64 + lane_in_group * 8);
        const float4 x1 = *reinterpret_cast<const float4*>(x + group * 64 + lane_in_group * 8 + 4);
        const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const std::int64_t i0 = static_cast<std::int64_t>(row0) * kGroups + group;
        const std::int64_t i1 = static_cast<std::int64_t>(row1) * kGroups + group;
        float w0[8], w1[8];
        q5_decode_eight(*reinterpret_cast<const std::uint32_t*>(codes + i0 * 32 + lane_in_group * 4),
                        high[i0 * 8 + lane_in_group],
                        *reinterpret_cast<const std::uint16_t*>(scales + i0 * 2), w0);
        q5_decode_eight(*reinterpret_cast<const std::uint32_t*>(codes + i1 * 32 + lane_in_group * 4),
                        high[i1 * 8 + lane_in_group],
                        *reinterpret_cast<const std::uint16_t*>(scales + i1 * 2), w1);
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            acc0 = fmaf(w0[item], values[item], acc0);
            acc1 = fmaf(w1[item], values[item], acc1);
        }
    }
    r0 = mk_warp_reduce_sum(acc0);
    r1 = mk_warp_reduce_sum(acc1);
}

__device__ __forceinline__ void w8_dot_fp32_row_x2(const std::uint8_t* codes,
                                                   const std::uint8_t* scales, int row0, int row1,
                                                   const float* x, float& r0, float& r1) {
    constexpr int kGroups = kIntermediate / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    for (int group = 0; group < kGroups; ++group) {
        const float xv           = x[group * 32 + lane];
        const std::int64_t i0    = static_cast<std::int64_t>(row0) * kGroups + group;
        const std::int64_t i1    = static_cast<std::int64_t>(row1) * kGroups + group;
        const float s0           = __half2float(__ushort_as_half(
            *reinterpret_cast<const std::uint16_t*>(scales + i0 * 2)));
        const float s1           = __half2float(__ushort_as_half(
            *reinterpret_cast<const std::uint16_t*>(scales + i1 * 2)));
        acc0 = fmaf(static_cast<float>(static_cast<std::int8_t>(codes[i0 * 32 + lane])) * s0, xv,
                    acc0);
        acc1 = fmaf(static_cast<float>(static_cast<std::int8_t>(codes[i1 * 32 + lane])) * s1, xv,
                    acc1);
    }
    r0 = mk_warp_reduce_sum(acc0);
    r1 = mk_warp_reduce_sum(acc1);
}

// dot_fp32_rows<Codec, 1> over K=512 activations (verbatim).
__device__ __forceinline__ float q5_dot_fp32_row(const std::uint8_t* codes,
                                                 const std::uint8_t* high,
                                                 const std::uint8_t* scales, int row,
                                                 const float* x) {
    constexpr int kGroups   = kIntermediate / 64;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int lane_group    = lane >> 3;
    const int lane_in_group = lane & 7;
    float acc               = 0.0f;
    for (int group_base = 0; group_base < kGroups; group_base += 4) {
        const int group = group_base + lane_group;
        const float4 x0 = *reinterpret_cast<const float4*>(x + group * 64 + lane_in_group * 8);
        const float4 x1 = *reinterpret_cast<const float4*>(x + group * 64 + lane_in_group * 8 + 4);
        const float values[8]          = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kGroups + group;
        const std::uint32_t packed =
            *reinterpret_cast<const std::uint32_t*>(codes + group_index * 32 + lane_in_group * 4);
        const std::uint8_t high_bits = high[group_index * 8 + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(scales + group_index * 2);
        float weights[8];
        q5_decode_eight(packed, high_bits, scale_bits, weights);
#pragma unroll
        for (int item = 0; item < 8; ++item) { acc = fmaf(weights[item], values[item], acc); }
    }
    return mk_warp_reduce_sum(acc);
}

__device__ __forceinline__ float w8_dot_fp32_row(const std::uint8_t* codes,
                                                 const std::uint8_t* scales, int row,
                                                 const float* x) {
    constexpr int kGroups = kIntermediate / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc             = 0.0f;
    for (int group = 0; group < kGroups; ++group) {
        const std::int64_t index = static_cast<std::int64_t>(row) * kGroups + group;
        const float scale        = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + index * 2)));
        const float w =
            static_cast<float>(static_cast<std::int8_t>(codes[index * 32 + lane])) * scale;
        acc = fmaf(w, x[group * 32 + lane], acc);
    }
    return mk_warp_reduce_sum(acc);
}

} // namespace moe

// Router scores: two 8-warp row groups per CTA, engine reduce order per row.
__device__ inline void mk_body_moe_d1(const MkInstr& instr, MkShared& shared) {
    const auto* x      = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* router = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    auto* scores       = static_cast<float*>(instr.out[0]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];

    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int group    = warp >> 3;
    const int warp_in8 = warp & 7;
    const std::int64_t row = row0 + group;

    if (group < 2 && row < row0 + rows && row < moe::kRouterRows) {
        constexpr int kSlice = moe::kHidden / 8;
        const auto* row_ptr  = router + row * moe::kHidden;
        float sum            = 0.0f;
        const int k          = warp_in8 * kSlice + lane * 8;
        sum += moe::mk_dot_bf16_eight(row_ptr + k, x + k);
        sum = mk_warp_reduce_sum(sum);
        if (lane == 0) { shared.d1.partial[group][warp_in8] = sum; }
    }
    __syncthreads();
    if ((warp == 0 || warp == 8) && group < 2 && row < row0 + rows && row < moe::kRouterRows) {
        float value = lane < 8 ? shared.d1.partial[group][lane] : 0.0f;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset, 8);
        }
        if (lane == 0) { scores[row] = value; }
    }
    __syncthreads();
}

// Top-8 select + softmax alpha + shared sigmoid scale: single warp (verbatim
// sparse_moe_select_top8_warp).
__device__ inline void mk_body_moe_d2(const MkInstr& instr, MkShared& shared) {
    const auto* scores = static_cast<const float*>(instr.ptr[0]);
    auto* shared_scale = const_cast<float*>(static_cast<const float*>(instr.ptr[1]));
    auto* ids          = static_cast<int*>(instr.out[0]);
    auto* alpha        = static_cast<float*>(instr.out[1]);
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    if (warp != 0) { return; }
    const int lane = static_cast<int>(threadIdx.x) & 31;

    struct Ranked {
        float value;
        int id;
        int origin;
    };
    const auto better = [](const Ranked& a, const Ranked& b) {
        return a.value > b.value || (a.value == b.value && a.id < b.id);
    };
    Ranked local[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id, lane};
    }
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const Ranked value = local[i];
        int position       = i;
        while (position > 0 && better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < moe::kTopK; ++rank) {
        Ranked candidate = cursor < 8 ? local[cursor] : Ranked{-CUDART_INF_F, 0x7fffffff, lane};
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            Ranked other;
            other.value  = __shfl_down_sync(0xffffffffu, candidate.value, offset);
            other.id     = __shfl_down_sync(0xffffffffu, candidate.id, offset);
            other.origin = __shfl_down_sync(0xffffffffu, candidate.origin, offset);
            if (better(other, candidate)) { candidate = other; }
        }
        candidate.value  = __shfl_sync(0xffffffffu, candidate.value, 0);
        candidate.id     = __shfl_sync(0xffffffffu, candidate.id, 0);
        candidate.origin = __shfl_sync(0xffffffffu, candidate.origin, 0);
        if (lane == 0) {
            ids[rank]                      = candidate.id;
            shared.d2.selected_logits[rank] = candidate.value;
        }
        if (lane == candidate.origin) { ++cursor; }
        __syncwarp();
    }
    float exponential = 0.0f;
    if (lane < moe::kTopK) {
        exponential = expf(shared.d2.selected_logits[lane] - shared.d2.selected_logits[0]);
    }
    float denominator = mk_warp_reduce_sum(exponential);
    denominator       = __shfl_sync(0xffffffffu, denominator, 0);
    if (lane < moe::kTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) {
        const float s = scores[moe::kExperts];
        *shared_scale = 1.0f / (1.0f + expf(-s));
    }
}

// gate_up + silu*up. Tasks = 9 paths x js j-values, distributed over ALL 16
// warps (each dot is warp-local, so any warp may own any task with identical
// FP order).
__device__ inline void mk_body_moe_d3(const MkInstr& instr) {
    const auto* x             = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* ids           = static_cast<const int*>(instr.ptr[1]);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(instr.ptr[2]);
    const auto* routed_scales = static_cast<const std::uint8_t*>(instr.ptr[4]);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(instr.ptr[5]);
    const auto* shared_scales = static_cast<const std::uint8_t*>(instr.ptr[6]);
    auto* act                 = static_cast<float*>(instr.out[0]);
    const int j0              = static_cast<int>(instr.dim[0]);
    const int js              = static_cast<int>(instr.dim[1]);

    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int tasks = (moe::kTopK + 1) * js;

    for (int task = warp; task < tasks; task += kMkThreads / 32) {
        const int path = task / js;
        const int j    = j0 + (task - path * js);
        // Warm the NEXT task's expert rows in L2 while this task's FMAs run:
        // d3 is latency-bound on random expert reads, not bandwidth-bound.
        const int next = task + kMkThreads / 32;
        if (instr.dim[7] != 0 && next < tasks) {
            const int npath = next / js;
            const int nj    = j0 + (next - npath * js);
            const std::int64_t nrow =
                npath < moe::kTopK
                    ? static_cast<std::int64_t>(ids[npath]) * (2 * moe::kIntermediate) + nj
                    : nj;
            const auto* base = npath < moe::kTopK
                                   ? routed_codes + nrow * (moe::kHidden / 64) * 32
                                   : shared_codes + nrow * (moe::kHidden / 32) * 32;
            if (lane < 16) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(base + lane * 128));
            }
        }
        float gate     = 0.0f;
        float up       = 0.0f;
        if (path < moe::kTopK) {
            const int expert   = ids[path];
            const int row_base = expert * (2 * moe::kIntermediate);
            moe::q4_dot_two_rows(routed_codes, routed_scales, row_base + j,
                                 row_base + moe::kIntermediate + j, x, gate, up);
        } else {
            moe::w8_dot_two_rows(shared_codes, shared_scales, j, moe::kIntermediate + j, x,
                                 gate, up);
        }
        if (lane == 0) {
            act[static_cast<std::int64_t>(path) * moe::kIntermediate + j] =
                moe::mk_silu(gate) * up;
        }
    }
}

// down + rank-ordered FP32 sum + residual. Tasks = 9 paths x 16 rows over all
// 16 warps; per-row path sums stay in fixed 0..8 order (single lane adds), so
// the epilogue matches the engine's deterministic rank-order accumulation.
__device__ inline void mk_body_moe_d4(const MkInstr& instr, MkShared& shared) {
    const auto* ids           = static_cast<const int*>(instr.ptr[0]);
    const auto* alpha         = static_cast<const float*>(instr.ptr[1]);
    const auto* shared_scale  = static_cast<const float*>(instr.ptr[2]);
    const auto* act           = static_cast<const float*>(instr.ptr[3]);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(instr.ptr[4]);
    const auto* routed_high   = static_cast<const std::uint8_t*>(instr.ptr[5]);
    const auto* routed_scales = static_cast<const std::uint8_t*>(instr.ptr[6]);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(instr.ptr[7]);
    const auto* shared_scales = static_cast<const std::uint8_t*>(instr.out[1]);
    auto* destination         = static_cast<__nv_bfloat16*>(instr.out[0]);
    const int row0            = static_cast<int>(instr.dim[0]);
    const int rows            = static_cast<int>(instr.dim[1]);   // <= 16

    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int tasks = (moe::kTopK + 1) * rows;

    for (int task = warp; task < tasks; task += kMkThreads / 32) {
        const int path = task / rows;
        const int rr   = task - path * rows;
        const int row  = row0 + rr;
        const int next = task + kMkThreads / 32;
        if (instr.dim[7] != 0 && next < tasks) {
            const int npath = next / rows;
            const int nrow  = row0 + (next - npath * rows);
            const auto* base =
                npath < moe::kTopK
                    ? routed_codes + (static_cast<std::int64_t>(ids[npath]) * moe::kHidden +
                                      nrow) *
                                         ((moe::kIntermediate / 64) * 32)
                    : shared_codes +
                          static_cast<std::int64_t>(nrow) * ((moe::kIntermediate / 32) * 32);
            if (lane < 2) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(base + lane * 128));
            }
        }
        float scaled;
        if (path < moe::kTopK) {
            const int expert = ids[path];
            const float dot  = moe::q5_dot_fp32_row(
                routed_codes, routed_high, routed_scales, expert * moe::kHidden + row,
                act + static_cast<std::int64_t>(path) * moe::kIntermediate);
            scaled = alpha[path] * dot;
        } else {
            const float dot = moe::w8_dot_fp32_row(
                shared_codes, shared_scales, row,
                act + static_cast<std::int64_t>(moe::kTopK) * moe::kIntermediate);
            scaled = *shared_scale * dot;
        }
        if (lane == 0) { shared.d4.paths[path][rr] = scaled; }
    }
    __syncthreads();
    for (int rr = warp; rr < rows; rr += kMkThreads / 32) {
        if (lane == 0) {
            const int row = row0 + rr;
            float value   = __bfloat162float(destination[row]);
#pragma unroll
            for (int path = 0; path < moe::kTopK + 1; ++path) {
                value += shared.d4.paths[path][rr];
            }
            destination[row] = __float2bfloat16_rn(value);
        }
    }
    __syncthreads();
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
    case MkOp::GdnGating:
        mk_body_gdn_gating(instr);
        return;
    case MkOp::MoeD1:
        mk_body_moe_d1(instr, shared);
        return;
    case MkOp::MoeD2:
        mk_body_moe_d2(instr, shared);
        return;
    case MkOp::MoeD3:
        mk_body_moe_d3(instr);
        return;
    case MkOp::MoeD4:
        mk_body_moe_d4(instr, shared);
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
    int enable_prefetch, unsigned long long* __restrict__ wait_ns,
    unsigned long long* __restrict__ exec_ns) {
    __shared__ MkShared shared;
    __shared__ std::uint32_t slice_shared;
    const MkStream stream = streams[blockIdx.x];
    for (std::uint32_t i = 0; i < stream.count; ++i) {
        const unsigned long long t_enter = wait_ns != nullptr ? clock64() : 0;
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
        const unsigned long long t_ready = wait_ns != nullptr ? clock64() : 0;
        // First pop; afterwards the done-post of slice N and the pop of slice N+1
        // share one thread-0 critical section (2 barriers per slice, not 3).
        if (threadIdx.x == 0) { slice_shared = atomicAdd(&counters[instr_s.task_counter], 1u); }
        __syncthreads();
        std::uint32_t idx = slice_shared;
        std::uint32_t done_local = 0;
        const bool batch_post    = instr_s.dim[6] != 0;
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
                if (batch_post) {
                    ++done_local;
                } else {
                    __threadfence();
                    if (instr_s.done_counter != kMkNone) {
                        atomicAdd(&counters[instr_s.done_counter], 1u);
                    }
                    if (instr_s.done2_counter != kMkNone && idx < instr_s.done2_limit) {
                        atomicAdd(&counters[instr_s.done2_counter], 1u);
                    }
                }
                slice_shared = atomicAdd(&counters[instr_s.task_counter], 1u);
            }
            __syncthreads();
            idx = slice_shared;
        }
        if (batch_post && threadIdx.x == 0 && done_local != 0 &&
            instr_s.done_counter != kMkNone) {
            __threadfence();
            atomicAdd(&counters[instr_s.done_counter], done_local);
        }
        if (wait_ns != nullptr && threadIdx.x == 0) {
            const unsigned long long t_done = clock64();
            atomicAdd(&wait_ns[i], t_ready - t_enter);
            atomicAdd(&exec_ns[i], t_done - t_ready);
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
