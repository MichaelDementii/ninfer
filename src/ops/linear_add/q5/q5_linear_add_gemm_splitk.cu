#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows           = 2048;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 48;
using ProjectionLauncher      = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Schedule heuristics transplanted from the W8 2048x4096 production path
// (W835bMtpProjectionGeometry / w8_linear_add_gemm_splitk.cu): 16 K-warps up to
// T=12, 8 beyond; the scale-access knob is unused (Q5 fetches its single g64
// scale per warp slab directly).
template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr int KWarps    = ActiveCols <= 12 ? 16 : 8;
    constexpr int MinBlocks = KWarps == 16 ? 1 : 2;
    constexpr auto ActivationCache =
        ActiveCols == 4 || (ActiveCols >= 27 && ActiveCols <= 40) ? Cache::cg : Cache::ca;
    using Schedule = W8SmallTMmaSchedule<KWarps, TileCols, MinBlocks,
                                         W8SmallTMmaScaleAccess::Direct, ActivationCache>;
    static_assert((kRows % kRowsPerCta) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), kRows};
    q5_small_t_mma_kernel<4096, ActiveCols, Schedule, W8ContiguousOutput>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kProjectionLaunchers =
    make_projection_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

} // namespace

void q5_linear_add_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                     cudaStream_t stream) {
    if (weight.k != 4096 || weight.padded_shape[1] != 4096 || residual_out.ne[0] != kRows) {
        throw std::invalid_argument("Q5 linear_add split-K MMA supports only exact 2048x4096");
    }
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("Q5 linear_add split-K MMA requires exact T=2..48");
    }
    kProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
