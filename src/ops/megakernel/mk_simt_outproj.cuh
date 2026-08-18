#pragma once

// Verbatim port of the author's T=1 out-proj route (w8_rowsplit_gemm_simt_kernel
// <W8RowSplitSimtSchedule, ColsPerTile=1, RowsPerCta=8, PipelineStages=2, Full,
// Residual>): one warp per output row, K staged through shared memory in
// 1024-value slabs with a cp.async double buffer, 4 consume phases per slab
// (lane owns [256c+8L, +8)), fp32 accumulate, warp-shuffle reduce, residual
// store. K=4096 => exactly 4 full slabs, no scalar tail.

#include "mk_core.cuh"

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::mk {

__device__ inline void mk_body_w8_simt_outproj(const MkInstr& instr, MkShared& shared) {
    const auto* x      = static_cast<const __nv_bfloat16*>(instr.ptr[0]);
    const auto* codes  = static_cast<const std::uint8_t*>(instr.ptr[1]);
    const auto* scales = static_cast<const std::uint8_t*>(instr.ptr[2]);
    auto* out          = static_cast<__nv_bfloat16*>(instr.out[0]);
    const std::int64_t row0 = instr.dim[0];
    const std::int64_t rows = instr.dim[1];

    constexpr int kK        = 4096;
    constexpr int kSlabs    = kK / 1024;
    constexpr int kGroups   = kK / 32;
    constexpr int kNibU4    = 64;
    constexpr int kScaleU32 = 16;
    constexpr int kStages   = 2;
    constexpr int kPrefetch = kStages - 1;

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = kMkThreads / 32;

    for (std::int64_t r = warp; r < rows; r += warps) {
        const std::int64_t row        = row0 + r;
        const std::uint8_t* code_row  = codes + row * kGroups * 32;
        const std::uint8_t* scale_row = scales + row * kGroups * 2;
        uint4* s_nib                  = &shared.simt.nib[warp][0][0];
        std::uint32_t* s_sc           = &shared.simt.sc[warp][0][0];

        const auto issue = [&](int slab, int stage) {
#pragma unroll
            for (int j = 0; j < kNibU4 / 32; ++j) {
                const int i = j * 32 + lane;
                pipe_copy<16>(
                    &s_nib[stage * kNibU4 + i],
                    code_row + static_cast<std::int64_t>(slab) * (kNibU4 * 16) + i * 16);
            }
            if (lane < kScaleU32) {
                pipe_copy<4>(
                    &s_sc[stage * kScaleU32 + lane],
                    scale_row + static_cast<std::int64_t>(slab) * (kScaleU32 * 4) + lane * 4);
            }
            pipe_commit();
        };

        float acc = 0.0f;
#pragma unroll
        for (int p = 0; p < kPrefetch; ++p) { issue(p, p); }

#pragma unroll 1
        for (int s = 0; s < kSlabs; ++s) {
            const int fetch = s + kPrefetch;
            if (fetch < kSlabs) {
                issue(fetch, fetch % kStages);
            } else {
                pipe_commit();
            }
            pipe_wait<kPrefetch>();
            __syncwarp();

            const int buf            = s % kStages;
            const uint4* nib         = &s_nib[buf * kNibU4];
            const std::uint32_t* sc  = &s_sc[buf * kScaleU32];
            const std::int64_t xslab = static_cast<std::int64_t>(s) * 1024;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint2 words = reinterpret_cast<const uint2*>(nib)[c * 32 + lane];
                const float scale = __half2float(__ushort_as_half(
                    reinterpret_cast<const std::uint16_t*>(sc)[c * 8 + (lane >> 2)]));
                float w[8];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const std::uint32_t word = (&words.x)[j];
                    w[4 * j + 0] =
                        static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
                    w[4 * j + 1] =
                        static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
                    w[4 * j + 2] =
                        static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
                    w[4 * j + 3] =
                        static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
                }
                const std::int64_t xoff = xslab + c * 256 + lane * 8;
                const uint4 xv          = *reinterpret_cast<const uint4*>(x + xoff);
                const float2 f0 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
                const float2 f1 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
                const float2 f2 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.z));
                const float2 f3 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.w));
                acc = fmaf(w[0], f0.x, acc);
                acc = fmaf(w[1], f0.y, acc);
                acc = fmaf(w[2], f1.x, acc);
                acc = fmaf(w[3], f1.y, acc);
                acc = fmaf(w[4], f2.x, acc);
                acc = fmaf(w[5], f2.y, acc);
                acc = fmaf(w[6], f3.x, acc);
                acc = fmaf(w[7], f3.y, acc);
            }
            __syncwarp();
        }

        acc = mk_warp_reduce_sum(acc);
        if (lane == 0) {
            out[row] = __float2bfloat16_rn(__bfloat162float(out[row]) + acc);
        }
    }
}

} // namespace ninfer::ops::mk
