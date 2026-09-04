#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
    int origin;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

// A warp-wide integer max is a warp-wide float max when the floats are mapped order-preservingly:
// non-negative values get their sign bit set, negative values are inverted. The mapping is exact
// and reversible, so nothing about the comparison changes -- unlike packing the value and the index
// into one 32-bit key, which would spend index bits out of the mantissa and move ties.
// Negative zero is folded onto positive zero first, because IEEE compares them equal and the
// selection breaks such a tie by index.
__device__ __forceinline__ std::uint32_t sparse_moe_order_key(float value) {
    const std::uint32_t bits = (value == 0.0f) ? 0u : __float_as_uint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float sparse_moe_order_value(std::uint32_t key) {
    return __uint_as_float((key & 0x80000000u) != 0u ? (key & 0x7fffffffu) : ~key);
}

__device__ __forceinline__ std::uint32_t sparse_moe_warp_max_u32(std::uint32_t value) {
    std::uint32_t result;
    asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}

__device__ __forceinline__ std::uint32_t sparse_moe_warp_min_u32(std::uint32_t value) {
    std::uint32_t result;
    asm("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
    return result;
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id, lane};
    }
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }

    // Each rank used to cost a five-step shuffle tournament plus a broadcast -- eight rounds of
    // eighteen dependent shuffles, on one warp, while the rest of the machine waits. Two redux.sync
    // reductions replace the whole round: one for the winning value, one for the lowest index
    // holding it, which is exactly the tie-break sparse_moe_ranked_better applies. Both broadcast
    // their result, so the reads below need no shuffle at all.
    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
        const bool present         = cursor < 8;
        const float value          = present ? local[cursor].value : -CUDART_INF_F;
        const std::uint32_t id     = present ? static_cast<std::uint32_t>(local[cursor].id)
                                             : 0x7fffffffu;
        const std::uint32_t key    = sparse_moe_order_key(value);
        const std::uint32_t best   = sparse_moe_warp_max_u32(key);
        const std::uint32_t winner = sparse_moe_warp_min_u32(key == best ? id : 0xffffffffu);
        if (lane == 0) {
            ids[rank]             = static_cast<int>(winner);
            selected_logits[rank] = sparse_moe_order_value(best);
        }
        if (key == best && id == winner) { ++cursor; }
        __syncwarp();
    }

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[kSparseMoeExperts]); }
}

} // namespace ninfer::ops::detail
