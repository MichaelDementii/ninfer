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

#ifdef NINFER_MK_ENGINE
// Engine-build-only dependencies for the fused gating transplant (the
// standalone mk_test compile has no include path to ops/common).
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#endif

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
// The engine's Offset-epilogue d=2048 route is rmsnorm_cta_bf16x2_kernel
// <Offset, 256, 6> (the d2048 512-thread kernel is Plain-only): 256 threads
// each accumulate FOUR pairs (t, t+256, t+512, t+768) sequentially, then a
// 256-wide block reduce (8 warp partials, width-8 shuffle tree). Threads
// 256..511 idle through the sum phase; the elementwise scale phase is
// thread-mapping independent, so all 512 threads share it.
__device__ inline void mk_body_rmsnorm2048(const MkInstr& instr, MkShared& shared) {
    const auto* x       = static_cast<const __nv_bfloat162*>(instr.ptr[0]);
    const auto* weight  = static_cast<const __nv_bfloat162*>(instr.ptr[1]);
    auto* out           = static_cast<__nv_bfloat162*>(instr.out[0]);
    const std::int64_t rows = instr.dim[0];
    const float eps         = __int_as_float(static_cast<int>(instr.dim[1]));

    constexpr int kCtaBlock    = 256;
    constexpr int kPairsPerRow = 1024;
    const int tid              = static_cast<int>(threadIdx.x);
    const int lane             = tid & 31;
    const int warp             = tid >> 5;

    for (std::int64_t row = 0; row < rows; ++row) {
        const std::int64_t row_base = row * kPairsPerRow;
        float sum                   = 0.0f;
        if (tid < kCtaBlock) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int pair  = tid + k * kCtaBlock;
                const float2 xf = __bfloat1622float2(x[row_base + pair]);
                sum += xf.x * xf.x + xf.y * xf.y;
            }
        }
        // block_reduce_sum<256>: warp partials from warps 0..7, width-8 tree.
        sum = mk_warp_reduce_sum(sum);
        if (lane == 0 && warp < 8) { shared.rms.warp_sums[warp] = sum; }
        __syncthreads();
        if (tid == 0) {
            // warp_reduce_sum<8> over the 8 partials, replicated serially with
            // the exact shuffle-tree grouping (offsets 4, 2, 1):
            // ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)).
            const float* s = shared.rms.warp_sums;
            const float t0 = (s[0] + s[4]) + (s[2] + s[6]);
            const float t1 = (s[1] + s[5]) + (s[3] + s[7]);
            shared.rms.inv = rsqrtf((t0 + t1) / 2048.0f + eps);
        }
        __syncthreads();
        const float inv = shared.rms.inv;

        // Scale phase: per-element independent math, all 512 threads.
        const int pair0 = tid;
        const int pair1 = tid + kMkThreads;
        const float2 x0 = __bfloat1622float2(x[row_base + pair0]);
        const float2 x1 = __bfloat1622float2(x[row_base + pair1]);
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
// __noinline__: own register allocation — inside the interpreter switch the body
// competes with every other class for the shared budget (122 regs) and ptxas
// stops hoisting the full 8-phase load volley the standalone 78-reg engine
// kernel enjoys; isolation restores the volley.
template <int K>
__device__ __forceinline__ void mk_w8_decode_rows(const __nv_bfloat16* __restrict__ x,
                                               const std::uint8_t* __restrict__ codes,
                                               const std::uint8_t* __restrict__ scales,
                                               __nv_bfloat16* __restrict__ out,
                                               std::int64_t row0, std::int64_t rows,
                                               bool residual) {
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

template <int K>
__device__ __forceinline__ void mk_body_w8_decode(const MkInstr& instr) {
    mk_w8_decode_rows<K>(static_cast<const __nv_bfloat16*>(instr.ptr[0]),
                         static_cast<const std::uint8_t*>(instr.ptr[1]),
                         static_cast<const std::uint8_t*>(instr.ptr[2]),
                         static_cast<__nv_bfloat16*>(instr.out[0]), instr.dim[0], instr.dim[1],
                         instr.dim[3] != 0);
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
        // Engine route for Gated d=128 is rmsnorm_warp_bf16x2 (the d128 kernel
        // is Plain-only): its per-lane sum adds PAIRWISE, sum += (x.x²+x.y²)
        // per k — not one flat four-term chain. Grouping matters at ULP.
        float sum = xf[0].x * xf[0].x + xf[0].y * xf[0].y;
        sum += xf[1].x * xf[1].x + xf[1].y * xf[1].y;
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
// __noinline__ for the same regalloc-isolation reason as mk_w8_decode_rows.
__device__ __forceinline__ void mk_w8_decode_conv_rows(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_state, __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ qc,
    __nv_bfloat16* __restrict__ kc, const std::int32_t* __restrict__ slots, std::int64_t row0,
    std::int64_t rows) {
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

    // Consecutive rows per warp (2w, 2w+1 for a 32-row slice): the warp's two
    // volleys stream adjacent 2KB code rows — DRAM row-buffer friendly, unlike
    // the strided (w, w+16) walk. Per-row math and order unchanged (bit-exact).
    const std::int64_t rows_per_warp = (rows + warps - 1) / warps;
    for (std::int64_t rr = 0; rr < rows_per_warp; ++rr) {
        const std::int64_t r = static_cast<std::int64_t>(warp) * rows_per_warp + rr;
        if (r >= rows) { break; }
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

__device__ __forceinline__ void mk_body_w8_decode_conv(const MkInstr& instr) {
    mk_w8_decode_conv_rows(
        static_cast<const __nv_bfloat16*>(instr.ptr[0]),
        static_cast<const std::uint8_t*>(instr.ptr[1]),
        static_cast<const std::uint8_t*>(instr.ptr[2]),
        static_cast<const __nv_bfloat16*>(instr.ptr[3]),
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(instr.ptr[4])),
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(instr.ptr[5])),
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(instr.ptr[6])),
        static_cast<__nv_bfloat16*>(instr.out[0]), static_cast<__nv_bfloat16*>(instr.out[1]),
        static_cast<const std::int32_t*>(instr.ptr[7]), instr.dim[0], instr.dim[1]);
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

        // dim[7] != 0: gating transform folded into this body (kills the gg tape
        // class and its arrival serialization). ptr[3] = ab (a[0..32), b[32..64)),
        // ptr[4] = A_log, ptr[7] = dt_bias; identical arithmetic to the gg class
        // on identical f32 inputs — bit-identical g/beta per head.
        float g_val;
        float beta_val;
        if (instr.dim[7] != 0) {
            const auto* dt_bias = static_cast<const float*>(instr.ptr[7]);
            const float av      = g_arr[head] + dt_bias[head];
            const float sp      = av > 20.0f ? av : log1pf(expf(av));
            g_val               = -expf(beta_arr[head]) * sp;
            beta_val            = 1.0f / (1.0f + expf(-g_arr[32 + head]));
        } else {
            g_val    = g_arr[head];
            beta_val = beta_arr[head];
        }
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
// __noinline__: the dot gets its own register allocation instead of sharing the
// interpreter switch's budget — with the full unroll every code/scale load for
// both rows is issued up front, which is what hides the cold-DRAM latency of
// random expert rows (the engine hides it with 2-3 CTAs/SM instead).
__device__ __noinline__ float2 q4_dot_two_rows(const std::uint8_t* codes,
                                               const std::uint8_t* scales, int row0, int row1,
                                               const __nv_bfloat16* x) {
    constexpr int kGroups   = kHidden / 64;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int lane_group    = lane >> 3;
    const int lane_in_group = lane & 7;
    float acc0              = 0.0f;
    float acc1              = 0.0f;
#pragma unroll
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
    return make_float2(mk_warp_reduce_sum(acc0), mk_warp_reduce_sum(acc1));
}

// dot_two_rows<W8Codec, 2048>: one value per lane per 32-group (verbatim).
// __noinline__ + unroll: same isolation rationale as q4_dot_two_rows; 64 groups
// unrolled is too many registers, so unroll by 8 (order of FMAs per acc chain is
// program order either way — bit-exact).
__device__ __noinline__ float2 w8_dot_two_rows(const std::uint8_t* codes,
                                               const std::uint8_t* scales, int row0, int row1,
                                               const __nv_bfloat16* x) {
    constexpr int kGroups = kHidden / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
#pragma unroll 8
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
    return make_float2(mk_warp_reduce_sum(acc0), mk_warp_reduce_sum(acc1));
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

__device__ __forceinline__ float q6_dot_fp32_row(const std::uint8_t* codes,
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
        const std::uint16_t high_bits =
            *reinterpret_cast<const std::uint16_t*>(high + group_index * 16 + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(scales + group_index * 2);
        float weights[8];
        q6_decode_eight(packed, high_bits, scale_bits, weights);
#pragma unroll
        for (int item = 0; item < 8; ++item) { acc = fmaf(weights[item], values[item], acc); }
    }
    return mk_warp_reduce_sum(acc);
}

__device__ __forceinline__ float w8_dot_fp32_row(const std::uint8_t* codes,
                                                 const std::uint8_t* scales, int row,
                                                 const float* x) {
    // Engine d4 W8 shared path = dot_fp32_rows' 16-lane PAIR branch (W8Codec has
    // no kPackedWord8): lane l < 16 owns k = group*32 + 2l, 2l+1, chained fmaf;
    // lanes 16..31 contribute exact zeros to the same full-width reduce. The
    // 32-lane single-value pattern gives a different per-lane partial split and
    // thus ULP-different sums — this was the MoE half's token divergence.
    constexpr int kGroups = kIntermediate / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc             = 0.0f;
    if (lane < 16) {
        for (int group = 0; group < kGroups; ++group) {
            const std::int64_t index = static_cast<std::int64_t>(row) * kGroups + group;
            const float scale        = __half2float(
                __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + index * 2)));
            const std::uint8_t* packed = codes + index * 32 + lane * 2;
            const float w0 = static_cast<float>(static_cast<std::int8_t>(packed[0])) * scale;
            const float w1 = static_cast<float>(static_cast<std::int8_t>(packed[1])) * scale;
            const float2 xv = *reinterpret_cast<const float2*>(x + group * 32 + lane * 2);
            acc             = fmaf(w0, xv.x, acc);
            acc             = fmaf(w1, xv.y, acc);
        }
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
// sparse_moe_select_top8_warp). When done2_counter is set (and counters passed),
// the ids[] array is published mid-body right after the selection loop: d3 needs
// ONLY ids, so it starts while this warp still computes softmax/shared-scale.
// done2_limit stays 0 so the interpreter's per-slice done2 path never fires.
__device__ inline void mk_body_moe_d2(const MkInstr& instr, MkShared& shared,
                                      std::uint32_t* counters) {
    const auto* scores = static_cast<const float*>(instr.ptr[0]);
    auto* shared_scale = const_cast<float*>(static_cast<const float*>(instr.ptr[1]));
    auto* ids          = static_cast<int*>(instr.out[0]);
    auto* alpha        = static_cast<float*>(instr.out[1]);
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    if (warp != 0) { return; }
    const int lane = static_cast<int>(threadIdx.x) & 31;

    // Same selection semantics (value desc, id asc), packed into one monotone
    // u64 key so each tournament step is one 64-bit shfl+compare instead of
    // three 32-bit shfls: hi32 = order-flipped float bits (desc), lo32 = ~id
    // (asc tiebreak). The owner lane is recoverable as id & 31, so no origin
    // field is needed. Identical comparator => identical ids and logits.
    const auto key_of = [](float v, int id) {
        std::uint32_t u = __float_as_uint(v);
        u               = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // asc in uint
        return (static_cast<unsigned long long>(u) << 32) |
               static_cast<std::uint32_t>(~id);
    };
    unsigned long long local[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int id = lane + item * 32;
        local[item]  = key_of(scores[id], id);
    }
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const unsigned long long value = local[i];
        int position                   = i;
        while (position > 0 && value > local[position - 1]) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < moe::kTopK; ++rank) {
        unsigned long long candidate =
            cursor < 8 ? local[cursor] : key_of(-CUDART_INF_F, 0x7fffffff);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other =
                __shfl_down_sync(0xffffffffu, candidate, offset);
            if (other > candidate) { candidate = other; }
        }
        candidate        = __shfl_sync(0xffffffffu, candidate, 0);
        const int win_id = static_cast<int>(~static_cast<std::uint32_t>(candidate));
        if (lane == 0) {
            std::uint32_t u = static_cast<std::uint32_t>(candidate >> 32);
            u               = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
            ids[rank]                       = win_id;
            shared.d2.selected_logits[rank] = __uint_as_float(u);
        }
        if (lane == (win_id & 31)) { ++cursor; }
        __syncwarp();
    }
    if (lane == 0 && counters != nullptr && instr.done2_counter != kMkNone) {
        __threadfence();   // release: publish the ids[] stores made by this lane
        atomicAdd(&counters[instr.done2_counter], 1u);
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

// gate_up + silu*up, engine-shaped locality: one slice = 16 CONSECUTIVE
// j-columns of ONE path, one warp per j — the CTA's 16 warps stream two
// contiguous 16KB windows (gate rows j0..j0+15, up rows 512+j0..) of a single
// expert instead of scattering across all nine. dim[3] selects the class:
//   dim[3]=1: SHARED path only (j = dim0+warp) — needs no ids, so the tape runs
//     it concurrently with d1/d2, streaming the shared expert's 2MB while the
//     router/top-8 would otherwise leave the bus idle.
//   dim[3]=0: routed paths, linear index over 8*512 j's (path = lin>>9).
// Per-output dot order identical to before — bit-exact.
__device__ inline void mk_body_moe_d3(const MkInstr& instr, MkShared& shared) {
    (void)shared;
    const auto* x = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* ids           = static_cast<const int*>(instr.ptr[1]);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(instr.ptr[2]);
    const auto* routed_scales = static_cast<const std::uint8_t*>(instr.ptr[4]);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(instr.ptr[5]);
    const auto* shared_scales = static_cast<const std::uint8_t*>(instr.ptr[6]);
    auto* act                 = static_cast<float*>(instr.out[0]);
    const int lin             = static_cast<int>(instr.dim[0]) +
                    (static_cast<int>(threadIdx.x) >> 5);   // one warp = one j
    const bool shared_only = instr.dim[3] != 0;

    float2 gate_up;
    int path;
    int j;
    if (shared_only) {
        path    = moe::kTopK;
        j       = lin;
        gate_up = moe::w8_dot_two_rows(shared_codes, shared_scales, j,
                                       moe::kIntermediate + j, x);
    } else {
        path               = lin >> 9;
        j                  = lin & (moe::kIntermediate - 1);
        const int expert   = ids[path];
        const int row_base = expert * (2 * moe::kIntermediate);
        gate_up = moe::q4_dot_two_rows(routed_codes, routed_scales, row_base + j,
                                       row_base + moe::kIntermediate + j, x);
    }
    if ((static_cast<int>(threadIdx.x) & 31) == 0) {
        act[static_cast<std::int64_t>(path) * moe::kIntermediate + j] =
            moe::mk_silu(gate_up.x) * gate_up.y;
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

    // Path-major rounds: with rows == 16, round k puts all 16 warps on PATH k,
    // warp w on row row0+w — the CTA streams one contiguous window of a single
    // expert's down matrix per round instead of scattering across nine. Per-row
    // dot order and the rank-ordered epilogue are unchanged (bit-exact).
    for (int path = 0; path <= moe::kTopK; ++path) {
        for (int rr = warp; rr < rows; rr += kMkThreads / 32) {
            const int row = row0 + rr;
            float scaled;
            if (path < moe::kTopK) {
                const int expert = ids[path];
                const float dot =
                    instr.dim[5] != 0
                        ? moe::q6_dot_fp32_row(
                              routed_codes, routed_high, routed_scales,
                              expert * moe::kHidden + row,
                              act + static_cast<std::int64_t>(path) * moe::kIntermediate)
                        : moe::q5_dot_fp32_row(
                              routed_codes, routed_high, routed_scales,
                              expert * moe::kHidden + row,
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

// ==== attention-island absorption bodies (all engine-route exact) ============

// w8_k2048 row dot (identical phase loop to mk_w8_decode_rows) + the engine's
// W8SplitOutput4<4096,512,4096,512> routing: q, k, gate, v in weight-row order.
__device__ __forceinline__ void mk_attn_qkv_rows(const __nv_bfloat16* __restrict__ x,
                                              const std::uint8_t* __restrict__ codes,
                                              const std::uint8_t* __restrict__ scales,
                                              __nv_bfloat16* __restrict__ q,
                                              __nv_bfloat16* __restrict__ k,
                                              __nv_bfloat16* __restrict__ gate,
                                              __nv_bfloat16* __restrict__ v, std::int64_t row0,
                                              std::int64_t rows) {
    constexpr int K                  = 2048;
    constexpr int kGroupsPerRow      = K / 32;
    constexpr int kValuesPerLane     = 8;
    constexpr int kValuesPerPhase    = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase    = kValuesPerPhase / 32;
    constexpr int kPhases            = K / kValuesPerPhase;
    constexpr unsigned kFullWarpMask = 0xffffffffu;

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = kMkThreads / 32;

    const std::int64_t rows_per_warp = (rows + warps - 1) / warps;
    for (std::int64_t rr = 0; rr < rows_per_warp; ++rr) {
        const std::int64_t r = static_cast<std::int64_t>(warp) * rows_per_warp + rr;
        if (r >= rows) { break; }
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
            const __nv_bfloat16 out = __float2bfloat16_rn(accumulator);
            if (row < 4096) {
                q[row] = out;
            } else if (row < 4608) {
                k[row - 4096] = out;
            } else if (row < 8704) {
                gate[row - 4608] = out;
            } else {
                v[row - 8704] = out;
            }
        }
    }
}

__device__ __forceinline__ void mk_body_attn_qkv(const MkInstr& instr) {
    mk_attn_qkv_rows(static_cast<const __nv_bfloat16*>(instr.ptr[0]),
                     static_cast<const std::uint8_t*>(instr.ptr[1]),
                     static_cast<const std::uint8_t*>(instr.ptr[2]),
                     static_cast<__nv_bfloat16*>(instr.out[0]),
                     static_cast<__nv_bfloat16*>(instr.out[1]),
                     const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(instr.ptr[5])),
                     const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(instr.ptr[6])),
                     instr.dim[0], instr.dim[1]);
}

// Engine route for Offset d=256 is rmsnorm_warp_bf16x2<Offset,512>: one warp per
// row, four bf16x2 pairs per lane (lane + k*32, k = 0..3), PAIRWISE += sums,
// inv = rsqrtf(sum/d + eps), out-of-place epilogue x*inv*(w+1).
__device__ inline void mk_body_norm_qk(const MkInstr& instr) {
    const auto* x      = static_cast<const __nv_bfloat162*>(instr.ptr[0]);
    const auto* weight = static_cast<const __nv_bfloat162*>(instr.ptr[1]);
    auto* out          = static_cast<__nv_bfloat162*>(instr.out[0]);
    const std::int64_t rows = instr.dim[0];
    const float eps         = __int_as_float(static_cast<int>(instr.dim[1]));

    constexpr int kPairs = 128;   // d = 256
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;

    for (std::int64_t row = warp; row < rows; row += kMkThreads / 32) {
        const std::int64_t row_base = row * kPairs;
        __nv_bfloat162 values[4];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int pair  = lane + k * 32;
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
        sum       = mk_warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum / 256.0f + eps) : 0.0f;
        inv       = __shfl_sync(0xffffffffu, inv, 0);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int pair  = lane + k * 32;
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            out[row_base + pair] = __floats2bfloat162_rn(xf.x * inv * (wf.x + 1.0f),
                                                         xf.y * inv * (wf.y + 1.0f));
        }
    }
}

// rope_fixed_kernel<Text1D, 16, 2> at T = 1: the engine's exact frequency table,
// sincosf per pair, in-place half-rotation of qn (16 heads) and kn (2 heads).
__device__ __constant__ float mk_rope_inv_freq[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

__device__ inline void mk_body_rope_qk(const MkInstr& instr, MkShared& shared) {
    const auto* positions = static_cast<const std::int32_t*>(instr.ptr[0]);
    auto* q               = static_cast<__nv_bfloat16*>(instr.out[0]);
    auto* k               = static_cast<__nv_bfloat16*>(instr.out[1]);

    constexpr int kHalf     = 32;
    constexpr int kHeadDim  = 256;
    constexpr int kHalfPair = kHalf / 2;
    float* cos_cache        = shared.rope.cs;
    float* sin_cache        = shared.rope.cs + kHalf;

    if (threadIdx.x < kHalf) {
        const int pair    = static_cast<int>(threadIdx.x);
        const float angle = static_cast<float>(positions[0]) * mk_rope_inv_freq[pair];
        sincosf(angle, &sin_cache[pair], &cos_cache[pair]);
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    float c0 = 0.0f, c1 = 0.0f, s0 = 0.0f, s1 = 0.0f;
    if (lane < kHalfPair) {
        const int pair = lane * 2;
        c0             = cos_cache[pair];
        c1             = cos_cache[pair + 1];
        s0             = sin_cache[pair];
        s1             = sin_cache[pair + 1];
    }
    for (int head = warp; head < 16 + 2; head += kMkThreads / 32) {
        __nv_bfloat16* data = head < 16 ? q : k;
        const int h         = head < 16 ? head : head - 16;
        if (lane >= kHalfPair) { continue; }
        auto* data2         = reinterpret_cast<__nv_bfloat162*>(data + h * kHeadDim);
        const float2 first  = __bfloat1622float2(data2[lane]);
        const float2 second = __bfloat1622float2(data2[lane + kHalfPair]);
        data2[lane] =
            __floats2bfloat162_rn(first.x * c0 - second.x * s0, first.y * c1 - second.y * s1);
        data2[lane + kHalfPair] =
            __floats2bfloat162_rn(second.x * c0 + first.x * s0, second.y * c1 + first.y * s1);
    }
    __syncthreads();
}

// sigmoid_gate_mul over bf16x8 packs (engine sigmoid_gate_mul_bf16x8_kernel):
// x[i] *= sigmoid(gate[i]), rounding per pair.
__device__ inline void mk_body_sig_gate_mul(const MkInstr& instr) {
    struct Pack {
        __nv_bfloat162 pair[4];
    };
    const auto* gate = static_cast<const Pack*>(instr.ptr[0]);
    auto* x          = static_cast<Pack*>(instr.out[0]);
    const std::int64_t packs = instr.dim[0] / 8;
    const auto sig           = [](float v) { return 1.0f / (1.0f + expf(-v)); };
    for (std::int64_t i = threadIdx.x; i < packs; i += kMkThreads) {
        const Pack gv = gate[i];
        Pack xv       = x[i];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float r0 = __low2float(xv.pair[p]) * sig(__low2float(gv.pair[p]));
            const float r1 = __high2float(xv.pair[p]) * sig(__high2float(gv.pair[p]));
            xv.pair[p]     = __floats2bfloat162_rn(r0, r1);
        }
        x[i] = xv;
    }
}

#ifdef NINFER_MK_ENGINE

namespace fg {
constexpr int kHeads       = 32;
constexpr int kHidden      = 2048;
constexpr int kBlockN      = 64;
constexpr int kBlockM      = 16;
constexpr int kBlockK      = 64;
constexpr int kStages      = 2;
constexpr int kSplitK      = 32;
constexpr int kWarps       = 8;    // author's 35B split-32 specialization
constexpr int kThreads     = kWarps * 32;
constexpr int kWarpN       = kBlockN / kWarps;   // 8
constexpr int kXStage      = kBlockN * kBlockK;  // 4096
constexpr int kWStage      = kBlockM * kBlockK;  // 1024
constexpr int kPartialRows = 2 * kHeads;         // 64
constexpr int kNormOffset  = kSplitK * kPartialRows;   // 2048

__device__ __forceinline__ int swz(int row, int col) {
    return (col & ~63) + ninfer::ops::detail::gemm_swz64(row, col & 63);
}
__device__ __forceinline__ void bar256() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
} // namespace fg

// Fused gating phase A: one slice = one (split, head-tile) CTA of the author's
// bf16_gdn_gating_proj_gemm_mma_kernel<Bf16Gdn35Geometry, SplitK=32,
// FullTokens=false, Warps=8, NormalizeInput=true> at t = 1. The smem swizzle,
// ldmatrix fragments and mma sequence are transplanted verbatim, so token 0's
// a/b partials and the per-split norm sums match the engine bit-for-bit.
// Only warps 0..7 participate (named barrier 1, 256 threads); __noinline__
// keeps the cp.async/ldmatrix machinery out of the interpreter's regalloc.
__device__ __noinline__ void mk_fused_gate_a_impl(const __nv_bfloat16* __restrict__ x,
                                                  const __nv_bfloat16* __restrict__ norm_w,
                                                  const __nv_bfloat16* __restrict__ a_w,
                                                  const __nv_bfloat16* __restrict__ b_w,
                                                  float* __restrict__ partial, int split,
                                                  int head0, __nv_bfloat16* __restrict__ smem) {
    using namespace fg;
    const int tid      = static_cast<int>(threadIdx.x);   // 0..255
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int kt_begin = split;   // kTilesPerSplit == 1 for 2048/64/32
    const int k0       = kt_begin * kBlockK;

    __nv_bfloat16* xs  = smem;
    __nv_bfloat16* aws = xs + kStages * kXStage;
    __nv_bfloat16* bws = aws + kStages * kWStage;

    // Author's norm slice: warp 0 of row-tile 0 contributes sum(x^2) over this
    // split's 64 elements (raw x, not the gain-scaled staging values).
    if (head0 == 0 && warp == 0) {
        const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
        float sum      = 0.0f;
        for (int pair = lane; pair < kBlockK / 2; pair += 32) {
            const float2 v = __bfloat1622float2(x2[kt_begin * (kBlockK / 2) + pair]);
            sum += v.x * v.x + v.y * v.y;
        }
        sum = mk_warp_reduce_sum(sum);
        if (lane == 0) { partial[kNormOffset + split] = sum; }
    }

    {   // stage_load(stage 0, kt_begin) — exact body at t = 1
        for (int vec = tid; vec < kBlockN * (kBlockK / 8); vec += kThreads) {
            const int token_local = vec / (kBlockK / 8);
            const int kk          = (vec - token_local * (kBlockK / 8)) * 8;
            __nv_bfloat16* dst = &xs[token_local * kBlockK + swz(token_local, kk)];
            if (token_local == 0) {
                const auto* source = reinterpret_cast<const __nv_bfloat162*>(&x[k0 + kk]);
                const auto* gain   = reinterpret_cast<const __nv_bfloat162*>(&norm_w[k0 + kk]);
                auto* target       = reinterpret_cast<__nv_bfloat162*>(dst);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 value  = __bfloat1622float2(source[pair]);
                    const float2 weight = __bfloat1622float2(gain[pair]);
                    target[pair] = __floats2bfloat162_rn(value.x * (1.0f + weight.x),
                                                         value.y * (1.0f + weight.y));
                }
            } else {
                *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
            }
        }
        const int weight_vecs = kBlockM * (kBlockK / 8);   // 128
        for (int all_vec = tid; all_vec < 2 * weight_vecs; all_vec += kThreads) {
            const bool is_b    = all_vec >= weight_vecs;
            const int vec      = is_b ? all_vec - weight_vecs : all_vec;
            const int row      = vec / (kBlockK / 8);
            const int kk       = (vec - row * (kBlockK / 8)) * 8;
            __nv_bfloat16* dst = (is_b ? bws : aws) + row * kBlockK + swz(row, kk);
            const __nv_bfloat16* weight = is_b ? b_w : a_w;
            cp_async<16, Cache::cg>(
                dst, &weight[static_cast<std::int64_t>(head0 + row) * kHidden + k0 + kk]);
        }
    }
    cp_commit();
    cp_commit();   // author's empty second-stage commit
    cp_wait<kStages - 1>();
    bar256();

    float a_acc[4] = {};
    float b_acc[4] = {};
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int x_rin    = lane & 7;
    const int x_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll
    for (int ki = 0; ki < kBlockK / 16; ++ki) {
        unsigned xf0;
        unsigned xf1;
        const int xrow = warp * kWarpN + x_rin;
        const int xcol = ki * 16 + x_koff;
        ldmatrix_x2(xf0, xf1, smem_addr(&xs[xrow * kBlockK + swz(xrow, xcol)]));
        unsigned af[4], bf[4];
        const int arow = a_rowoff;
        const int acol = ki * 16 + a_coloff;
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(&aws[arow * kBlockK + swz(arow, acol)]));
        ldmatrix_x4(bf[0], bf[1], bf[2], bf[3],
                    smem_addr(&bws[arow * kBlockK + swz(arow, acol)]));
        mma_bf16(a_acc[0], a_acc[1], a_acc[2], a_acc[3], af[0], af[1], af[2], af[3], xf0, xf1);
        mma_bf16(b_acc[0], b_acc[1], b_acc[2], b_acc[3], bf[0], bf[1], bf[2], bf[3], xf0, xf1);
    }

    // t == 1: only token-column 0 stores land (warp 0, lid 0 lanes).
    const int col0 = warp * kWarpN + 2 * lid;
    if (col0 == 0) {
        const int row0                              = head0 + gid;
        const int row1                              = row0 + 8;
        float* base                                 = partial + split * kPartialRows;
        base[row0]                                  = a_acc[0];
        base[kHeads + row0]                         = b_acc[0];
        base[row1]                                  = a_acc[2];
        base[kHeads + row1]                         = b_acc[2];
    }
    bar256();   // slice reuse: next pop's stage_load must not race these reads
}

__device__ inline void mk_body_fused_gate_a(const MkInstr& instr, MkShared& shared) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (warp >= fg::kWarps) { return; }
    const int lin = static_cast<int>(instr.dim[0]);
    mk_fused_gate_a_impl(static_cast<const __nv_bfloat16*>(instr.ptr[0]),
                         static_cast<const __nv_bfloat16*>(instr.ptr[1]),
                         static_cast<const __nv_bfloat16*>(instr.ptr[2]),
                         static_cast<const __nv_bfloat16*>(instr.ptr[3]),
                         static_cast<float*>(instr.out[0]), lin & 31, (lin >> 5) * 16,
                         shared.fg.stage);
}

// Phase B: the author's post-grid-sync epilogue — ordered split reduction for
// the norm (s = 0..31), h = x * inv * (1 + w) elementwise, then a/b ordered
// reduction scaled by inv and the gating transform. Bit-identical to the
// engine's cooperative handoff.
__device__ inline void mk_body_fused_gate_b(const MkInstr& instr) {
    const auto* x       = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* norm_w  = static_cast<const __nv_bfloat16*>(instr.ptr[1]);
    const auto* partial = static_cast<const float*>(instr.ptr[2]);
    const auto* a_log   = static_cast<const float*>(instr.ptr[3]);
    const auto* dt_bias = static_cast<const float*>(instr.ptr[4]);
    auto* beta          = const_cast<float*>(static_cast<const float*>(instr.ptr[5]));
    auto* h             = static_cast<__nv_bfloat16*>(instr.out[0]);
    auto* g             = static_cast<float*>(instr.out[1]);
    const float eps     = __int_as_float(static_cast<int>(instr.dim[1]));

    const float* norm_partial = partial + fg::kNormOffset;
    float sum                 = 0.0f;
#pragma unroll
    for (int s = 0; s < fg::kSplitK; ++s) { sum += norm_partial[s]; }
    const float inv = rsqrtf(sum / static_cast<float>(fg::kHidden) + eps);

    for (int i = static_cast<int>(threadIdx.x); i < fg::kHidden; i += kMkThreads) {
        const float value  = __bfloat162float(x[i]);
        const float weight = __bfloat162float(norm_w[i]);
        h[i]               = __float2bfloat16_rn(value * inv * (1.0f + weight));
    }
    for (int i = static_cast<int>(threadIdx.x); i < fg::kHeads; i += kMkThreads) {
        float av = 0.0f;
        float bv = 0.0f;
#pragma unroll
        for (int s = 0; s < fg::kSplitK; ++s) {
            av += partial[s * fg::kPartialRows + i];
            bv += partial[s * fg::kPartialRows + fg::kHeads + i];
        }
        av *= inv;
        bv *= inv;
        g[i]    = -expf(a_log[i]) * softplus(av + dt_bias[i]);
        beta[i] = sigmoid(bv);
    }
}

#endif // NINFER_MK_ENGINE

__device__ __forceinline__ void mk_execute(const MkInstr& instr, MkShared& shared,
                                           std::uint32_t* counters) {
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
        mk_body_moe_d2(instr, shared, counters);
        return;
    case MkOp::MoeD3:
        mk_body_moe_d3(instr, shared);
        return;
    case MkOp::MoeD4:
        mk_body_moe_d4(instr, shared);
        return;
    case MkOp::FusedGateA:
#ifdef NINFER_MK_ENGINE
        mk_body_fused_gate_a(instr, shared);
#endif
        return;
    case MkOp::FusedGateB:
#ifdef NINFER_MK_ENGINE
        mk_body_fused_gate_b(instr);
#endif
        return;
    case MkOp::AttnQkv:
        mk_body_attn_qkv(instr);
        return;
    case MkOp::NormQK:
        mk_body_norm_qk(instr);
        return;
    case MkOp::RopeQK:
        mk_body_rope_qk(instr, shared);
        return;
    case MkOp::SigGateMul:
        mk_body_sig_gate_mul(instr);
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
    mk_execute(local, shared, nullptr);
}

// ---- the interpreter -------------------------------------------------------

// While lane 0 spins on dependency counters, the other threads pull one slice's
// worth of this class's WEIGHT bytes into L2 — weights are immutable, so
// prefetching ahead of the data dependency is always safe, and the DRAM fetch
// overlaps the wait instead of following it. With dynamic slice popping the
// exact future assignment is unknown; the CTA's grid rank approximates its
// first-wave slice.
__device__ __forceinline__ void mk_prefetch_slice(const MkInstr& instr, unsigned rank) {
    if (rank >= instr.slice_count) { return; }
    if (instr.op == MkOp::FusedGateA) {
        // Warm this slice's a/b weight tile (16 rows x 64 k each) while waiting
        // on the previous layer's tail — hides the tile's DRAM latency.
        const int lin   = static_cast<int>(instr.dim[0]) + static_cast<int>(rank);
        const int split = lin & 31;
        const int head0 = (lin >> 5) * 16;
        const auto* aw  = static_cast<const char*>(instr.ptr[2]);
        const auto* bw  = static_cast<const char*>(instr.ptr[3]);
        const int row   = static_cast<int>(threadIdx.x) & 15;
        const std::int64_t off =
            (static_cast<std::int64_t>(head0 + row) * 2048 + split * 64) * 2;
        if (threadIdx.x < 16) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(aw + off));
        } else if (threadIdx.x < 32) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(bw + off));
        }
        return;
    }
    if (instr.op != MkOp::W8DecodeK && instr.op != MkOp::W8DecodeConv &&
        instr.op != MkOp::AttnQkv) {
        return;
    }
    const auto* codes        = static_cast<const char*>(instr.ptr[1]);
    const std::int64_t row0  = instr.dim[0] + static_cast<std::int64_t>(rank) * instr.dim[1];
    const std::int64_t base  = row0 * instr.dim[2];
    const std::int64_t bytes = instr.dim[1] * instr.dim[2];
    for (std::int64_t off = static_cast<std::int64_t>(threadIdx.x) * 128; off < bytes;
         off += static_cast<std::int64_t>(kMkThreads) * 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(codes + base + off));
    }
}

__device__ __forceinline__ unsigned long long mk_globaltimer() {
    unsigned long long gt;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt));
    return gt;
}

// Prof-v2 wall-span attribution (all pointers null outside prof mode):
//   span_min/span_max: per tape index, first work-start / last slice-end in
//   globaltimer ns (reset to ~0/0 each round by begin_round's memset nodes, so
//   after a run they hold the LAST round's timeline — graph replays included).
//   seg_stamp: [seg] = min kernel entry, [32+seg] = max kernel exit per segment.
__global__ __launch_bounds__(kMkThreads, 1) void mk_interpreter_kernel(
    const MkStream* __restrict__ streams, std::uint32_t* __restrict__ counters,
    int enable_prefetch, unsigned long long* __restrict__ wait_ns,
    unsigned long long* __restrict__ exec_ns, unsigned long long* __restrict__ span_min,
    unsigned long long* __restrict__ span_max, unsigned long long* __restrict__ seg_stamp,
    int seg_idx) {
    __shared__ MkShared shared;
    __shared__ std::uint32_t slice_shared;
    if (seg_stamp != nullptr && threadIdx.x == 0) {
        atomicMin(&seg_stamp[seg_idx], mk_globaltimer());
    }
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
        if (i == 0) {
            // PDL: when this segment is launched as a programmatic dependent of
            // the attention island's last kernel (which triggers at its head),
            // everything above — tape broadcast, first-class weight prefetch —
            // overlapped the island. Data reads gate here on its full completion.
            // No-op for eager launches without upstream dependencies.
            cudaGridDependencySynchronize();
        }
        mk_wait_phase(instr_s, counters);
        const unsigned long long t_ready = wait_ns != nullptr ? clock64() : 0;
        unsigned long long gt_ready      = 0;
        if (span_min != nullptr && threadIdx.x == 0) { gt_ready = mk_globaltimer(); }
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
            mk_execute(local, shared, counters);
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
            if (span_min != nullptr) {
                atomicMin(&span_min[i], gt_ready);
                atomicMax(&span_max[i], mk_globaltimer());
            }
        }
    }
    if (seg_stamp != nullptr && threadIdx.x == 0) {
        atomicMax(&seg_stamp[32 + seg_idx], mk_globaltimer());
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
