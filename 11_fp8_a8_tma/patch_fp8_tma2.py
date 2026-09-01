#!/usr/bin/env python3
"""П4, часть 2: маршрут TMA во всех семействах операций, которые реально работают в префилле.

Плоский `linear` в префилле не вызывается ни разу: вход GDN идёт через `gdn_input_proj`, вход
внимания через `attn_input_proj`, выходы GDN и MLP-down через `linear_add`, а MLP gate/up через
`linear_swiglu`. Все пять инстанцируют одно и то же `fp8_mma_kernel`, отличаясь только политикой
эпилога и приёмника, поэтому маршрут добавляется одной общей функцией запуска.

`linear_swiglu` здесь не трогается: он использует `RowPolicy` и парную запись строк, которых у
TMA-ядра пока нет.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

# --- общая функция запуска ---------------------------------------------------------------------
patch('src/ops/linear/fp8/fp8_a8_tma.cuh', [
    ("""} // namespace ninfer::ops::detail""",
     """// One launch path for every Op family: they differ only in the epilogue and output policies.
template <class Geometry, class Schedule, class Epilogue, class Output>
void fp8_a8_tma_launch(const std::uint8_t* activation_codes, const float* activation_scales,
                       const std::uint8_t* weight_codes, const __nv_bfloat16* weight_scales,
                       std::int32_t tokens, Epilogue epilogue, Output output,
                       cudaStream_t stream) {
    constexpr std::size_t kSharedBytes = sizeof(Fp8A8TmaSharedStorage<Schedule>);
    static_assert(kSharedBytes <= 99 * 1024);
    static const cudaError_t attribute =
        cudaFuncSetAttribute(fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(kSharedBytes));
    if (attribute != cudaSuccess) { throw std::runtime_error("fp8 TMA shared memory attribute"); }

    const Fp8A8TmaDescriptors descriptors =
        make_fp8_a8_tma_descriptors<Geometry, Schedule::kBlockM, Schedule::kBlockN>(
            activation_codes, weight_codes, tokens);
    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, activation_scales,
                                                               weight_scales, epilogue, output);
}

} // namespace ninfer::ops::detail"""),
])

# --- вход GDN ----------------------------------------------------------------------------------
patch('src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu', [
    ("""    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, qkv, z, workspace, x.ne[1], stream);""",
     """    launch_fp8_a8_quantize(x, weight, workspace, stream);
    using TmaSchedule = typename Fp8LinearA8TmaSchedule<Geometry>::Type;
    if (fp8_a8_tma_applies<Geometry, TmaSchedule>(x.ne[1])) {
        const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                       static_cast<__nv_bfloat16*>(z.data)};
        fp8_a8_tma_launch<Geometry, TmaSchedule>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), x.ne[1], Fp8IdentityEpilogue{},
            output, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, qkv, z, workspace, x.ne[1], stream);"""),
])

# --- вход внимания -----------------------------------------------------------------------------
patch('src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu', [
    ("""    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, q, gate, k, v, workspace, x.ne[1], stream);""",
     """    launch_fp8_a8_quantize(x, weight, workspace, stream);
    using TmaSchedule = typename Fp8LinearA8TmaSchedule<Geometry>::Type;
    if (fp8_a8_tma_applies<Geometry, TmaSchedule>(x.ne[1])) {
        const Fp8AttentionInputOutput output{
            static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(v.data),
        };
        fp8_a8_tma_launch<Geometry, TmaSchedule>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), x.ne[1], Fp8IdentityEpilogue{},
            output, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, q, gate, k, v, workspace, x.ne[1], stream);"""),
])

# --- выходы GDN и MLP-down ---------------------------------------------------------------------
patch('src/ops/linear_add/fp8/fp8_linear_add_a8.cu', [
    ("""template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    if ((tokens % Schedule::kBlockTokens) == 0) {""",
     """template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    using Schedule    = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    using TmaSchedule = typename Fp8LinearA8TmaSchedule<Geometry>::Type;
    if (fp8_a8_tma_applies<Geometry, TmaSchedule>(tokens)) {
        auto* output = static_cast<__nv_bfloat16*>(residual.data);
        fp8_a8_tma_launch<Geometry, TmaSchedule>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens,
            Fp8AddResidualEpilogue{output, Geometry::kOutputRows},
            Fp8ContiguousOutput{output, Geometry::kOutputRows}, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((tokens % Schedule::kBlockTokens) == 0) {"""),
])
print('маршрут TMA заведён в четыре семейства')
