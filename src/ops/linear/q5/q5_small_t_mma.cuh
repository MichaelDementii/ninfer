#pragma once

// Q5G64 RowSplit exact-small-T MMA core: the W8G32 kernel from
// ops/linear/w8/w8_small_t_mma.cuh retargeted at the Q5 plane layout.
//
// K-split warps cooperatively own one 16-row output tile; each warp evaluates a
// disjoint 64-wide K slice (= exactly one Q5 quant group), then the CTA reduces FP32
// partials in shared memory. Weights decode nib+high -> signed 5-bit -> bf16 pairs
// feeding m16n8k16 MMAs; the single per-group fp16 scale folds into the FP32
// accumulator once per group, so the tensor cores run on unscaled integer values
// exactly as in the W8 kernel.
//
// Plane addressing (Q5RowSplitStorage): per row K/2 nibble bytes, K/8 high bytes,
// K/32 scale bytes; within a group, byte b holds elements (2b, 2b+1) whose 5th bits
// sit at (2(b&3), 2(b&3)+1) of high byte (b>>2).
//
// Shared staging keeps the W8 activation path (swizzled, ldmatrix-fed) untouched.
// The weight planes are small enough that a +16B row pad staggers banks across the
// 8 fragment rows a thread quad walks, replacing the W8 chunk-xor swizzle.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Assemble one (q0, q1) code pair from a nibble byte and its two high bits into the
// u32 shape w8_small_t_bf16_pair_from_s8 expects: bytes 0/1 carry the values as
// two's-complement s8.
__device__ __forceinline__ unsigned q5_small_t_s8_pair(unsigned nib_byte, unsigned high2) {
    unsigned v = (nib_byte & 0x0fu) | ((nib_byte & 0xf0u) << 4) | ((high2 & 1u) << 4) |
                 ((high2 & 2u) << 11);
    v |= (v & 0x1010u) * 0x0Eu; // per-byte sign-extend of the 5-bit values to s8
    return v;
}

template <int Hidden, int ActiveCols, class Schedule, class Output>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q5_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ nib,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    Output output) {
    constexpr int kHidden       = Hidden;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kMmaRows      = Schedule::kRowsPerCta;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kSlabK        = kWarps * kTileK;
    constexpr int kSlabs        = kHidden / kSlabK;
    constexpr int kTileCols     = Schedule::kTileTokens;
    constexpr int kGroupsPerRow = kHidden / Q5RowSplitStorage::kGroupK;
    static_assert(kTileK == Q5RowSplitStorage::kGroupK,
                  "one warp slab slice must equal one Q5 quant group");
    static_assert((kHidden % kSlabK) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols);
    constexpr int kNt        = kTileCols / 8;
    constexpr unsigned kMask = 0xffffffffu;

    constexpr int kNibRowBytes  = kWarps * Q5RowSplitStorage::kCodeBytesPerGroup + 16;
    constexpr int kHighRowBytes = kWarps * Q5RowSplitStorage::kHighBytesPerGroup + 16;

    union SharedStorage {
        struct {
            std::uint8_t nib[kMmaRows][kNibRowBytes];
            std::uint8_t high[kMmaRows][kHighRowBytes];
            __nv_bfloat16 activations[kWarps][kTileCols * kTileK];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& nib_shared  = shared.staging.nib;
    auto& high_shared = shared.staging.high;
    auto& b_shared    = shared.staging.activations;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;

    const int cta_row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int slab_k0) {
        constexpr bool kPaddedStage =
            Schedule::kActivationStage == W8SmallTMmaActivationStage::PaddedZero;
        constexpr int kStageCols     = kPaddedStage ? kTileCols : ActiveCols;
        constexpr int kItemsPerSplit = kStageCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)];
            if constexpr (!kPaddedStage || ActiveCols == kTileCols) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(col) * kHidden + slab_k0 + warp * kTileK +
                            k8 * 8]);
            } else {
                const int source_col = col < ActiveCols ? col : 0;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(source_col) * kHidden + slab_k0 + warp * kTileK +
                       k8 * 8],
                    col < ActiveCols ? 16 : 0);
            }
        }
    };

    const auto stage_weights = [&](int slab) {
        constexpr int kNibChunks  = kWarps * Q5RowSplitStorage::kCodeBytesPerGroup / 16;
        constexpr int kHighChunks = kWarps * Q5RowSplitStorage::kHighBytesPerGroup / 16;
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = cta_row0 + row;
            for (int chunk = lane; chunk < kNibChunks; chunk += 32) {
                cp_async<16, Schedule::kWeightCache>(
                    &nib_shared[row][chunk * 16],
                    nib + static_cast<std::int64_t>(weight_row) * (kHidden / 2) +
                        slab * (kWarps * Q5RowSplitStorage::kCodeBytesPerGroup) + chunk * 16);
            }
            for (int chunk = lane; chunk < kHighChunks; chunk += 32) {
                cp_async<16, Schedule::kWeightCache>(
                    &high_shared[row][chunk * 16],
                    high + static_cast<std::int64_t>(weight_row) * (kHidden / 8) +
                        slab * (kWarps * Q5RowSplitStorage::kHighBytesPerGroup) + chunk * 16);
            }
        }
    };

    const int b_rin  = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    float acc[kNt][4];
#pragma unroll
    for (int ni = 0; ni < kNt; ++ni) {
        acc[ni][0] = 0.0f;
        acc[ni][1] = 0.0f;
        acc[ni][2] = 0.0f;
        acc[ni][3] = 0.0f;
    }

    stage_weights(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int slab = 0; slab < kSlabs; ++slab) {
        const int group = slab * kWarps + k_split;

        unsigned lane_scale = 0;
        if (lid < 2) {
            const int scale_row = cta_row0 + gid + lid * 8;
            lane_scale          = *reinterpret_cast<const std::uint16_t*>(
                scales + (static_cast<std::int64_t>(scale_row) * kGroupsPerRow + group) * 2);
        }
        const float top_scale = __half2float(__ushort_as_half(
            static_cast<unsigned short>(__shfl_sync(kMask, lane_scale, lane & ~3))));
        const float bot_scale = __half2float(__ushort_as_half(
            static_cast<unsigned short>(__shfl_sync(kMask, lane_scale, (lane & ~3) + 1))));

        float group_acc[kNt][4];
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            group_acc[ni][0] = 0.0f;
            group_acc[ni][1] = 0.0f;
            group_acc[ni][2] = 0.0f;
            group_acc[ni][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int nib_off  = k_split * Q5RowSplitStorage::kCodeBytesPerGroup + ks * 8 + lid;
            const int high_off = k_split * Q5RowSplitStorage::kHighBytesPerGroup + 2 * ks;
            const unsigned ht =
                *reinterpret_cast<const unsigned short*>(&high_shared[gid][high_off]);
            const unsigned hb =
                *reinterpret_cast<const unsigned short*>(&high_shared[gid + 8][high_off]);
            const unsigned sh  = 2u * static_cast<unsigned>(lid);
            const unsigned af0 = w8_small_t_bf16_pair_from_s8(
                q5_small_t_s8_pair(nib_shared[gid][nib_off], (ht >> sh) & 3u));
            const unsigned af1 = w8_small_t_bf16_pair_from_s8(
                q5_small_t_s8_pair(nib_shared[gid + 8][nib_off], (hb >> sh) & 3u));
            const unsigned af2 = w8_small_t_bf16_pair_from_s8(
                q5_small_t_s8_pair(nib_shared[gid][nib_off + 4], (ht >> (8u + sh)) & 3u));
            const unsigned af3 = w8_small_t_bf16_pair_from_s8(
                q5_small_t_s8_pair(nib_shared[gid + 8][nib_off + 4], (hb >> (8u + sh)) & 3u));
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
                unsigned bf0, bf1;
                const int br = ni * 8 + b_rin;
                ldmatrix_x2(
                    bf0, bf1,
                    smem_addr(&b_shared[k_split][br * kTileK +
                                                 w8_small_t_swizzle_64(br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            acc[ni][0] = fmaf(group_acc[ni][0], top_scale, acc[ni][0]);
            acc[ni][1] = fmaf(group_acc[ni][1], top_scale, acc[ni][1]);
            acc[ni][2] = fmaf(group_acc[ni][2], bot_scale, acc[ni][2]);
            acc[ni][3] = fmaf(group_acc[ni][3], bot_scale, acc[ni][3]);
        }

        if (slab + 1 < kSlabs) {
            __syncthreads();
            stage_weights(slab + 1);
            stage_x((slab + 1) * kSlabK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                      make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            const float4 partner =
                load_vec<float4>(partial + (((warp + 1) * kNt + ni) * 32 + lane) * 4);
            acc[ni][0] += partner.x;
            acc[ni][1] += partner.y;
            acc[ni][2] += partner.z;
            acc[ni][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                          make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
        const W8OutputTile output_tile = output.tile(cta_row0);
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            float4 sum = make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + ni) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0   = ni * 8 + 2 * lid;
            const auto store = [&](int row, int col, float value) {
                __nv_bfloat16* destination = output_tile.at(row, col);
                value += __bfloat162float(*destination);
                *destination = __float2bfloat16_rn(value);
            };
            if (col0 < ActiveCols) {
                store(cta_row0 + gid, col0, sum.x);
                store(cta_row0 + gid + 8, col0, sum.z);
            }
            if (col0 + 1 < ActiveCols) {
                store(cta_row0 + gid, col0 + 1, sum.y);
                store(cta_row0 + gid + 8, col0 + 1, sum.w);
            }
        }
    }
}

} // namespace ninfer::ops::detail
