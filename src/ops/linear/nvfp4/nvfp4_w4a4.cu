#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

using M32N64            = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           Nvfp4ScaleLayout scale_layout, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    const int blocks          = (tasks + kThreads - 1) / kThreads;
    if (scale_layout == Nvfp4ScaleLayout::Tiled) {
        nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::Tiled>
            <<<blocks, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              workspace.codes, workspace.scales, tokens,
                                              weight.input_scale_divisor);
    } else {
        nvfp4_w4a4_quantize_kernel<ActivationGeometry><<<blocks, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            weight.input_scale_divisor);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    constexpr bool kResidualGeometry = std::is_same_v<Geometry, Nvfp4Residual6144Geometry> ||
                                       std::is_same_v<Geometry, Nvfp4Residual17408Geometry>;
    if (nvfp4_w4a4_tma_route(tokens)) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear(
            resolve_nvfp4_problem(Geometry::kOutputRows, Geometry::kInputRows), workspace.codes,
            workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Geometry, M32N128>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 128) {
        if constexpr (kResidualGeometry) {
            launch_gemm<Geometry, M32N128>(weight, out, workspace, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, out, workspace, tokens, stream);
        }
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 512) {
        if constexpr (Geometry::kOutputRows == Nvfp4GdnInputGeometry::kOutputRows) {
            launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, out, workspace, tokens, stream);
        }
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, out, workspace, tokens, stream);
    }
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                Nvfp4ScaleLayout scale_layout, cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    // The tiled layout is a bijection onto the scale plane only for a whole number of token tiles.
    // Every route that asks for it admits only such widths; check it here, where the layout is
    // acted on, rather than trust each caller's own predicate.
    if (scale_layout == Nvfp4ScaleLayout::Tiled && (x.ne[1] % kNvfp4TmaBlockM) != 0) {
        throw std::invalid_argument(
            "nvfp4 W4A4 tiled activation scales require a whole number of token tiles");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, scale_layout,
                                                           stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, scale_layout,
                                                           stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, scale_layout,
                                                            stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, nvfp4_w4a4_scale_layout(x.ne[1]), stream);
    const std::int32_t tokens = x.ne[1];
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launch_problem<Nvfp4AttnInputGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_problem<Nvfp4GdnInputGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_problem<Nvfp4MlpGateUpGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Residual6144Geometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Residual17408Geometry>(weight, out, workspace, tokens, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
