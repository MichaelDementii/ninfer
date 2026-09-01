#!/usr/bin/env python3
"""П4, часть 3: парная запись строк, то есть маршрут TMA для `linear_swiglu`.

Слитый SwiGLU считает в одном CTA обе ветви: строки [0, BN/2) — гейт, строки [BN/2, BN) — up, и
вторая ветвь лежит в весах со смещением `kIntermediate`. У `cp.async`-ядра это уже сделано через
`RowPolicy` и флаг `PairRows`; здесь то же самое добавлено TMA-ядру.

Разница ровно одна: тензорный дескриптор весов описывает коробку в BN/2 строк, и продюсер выдаёт
на стадию **два** запроса вместо одного — по одному на ветвь. Число байт транзакции то же, эпилог
складывает пару и пишет её `store_pair_vector`.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

patch('src/ops/linear/fp8/fp8_a8_tma.cuh', [
    # 1. дескриптор весов: коробка в половину строк, когда ветвей две
    ("""template <class Geometry, int BlockM, int BlockN>
Fp8A8TmaDescriptors make_fp8_a8_tma_descriptors(const std::uint8_t* activation_codes,
                                                const std::uint8_t* weight_codes,
                                                std::int32_t tokens) {
    constexpr std::uint32_t kRowBytes = 64;""",
     """template <class Geometry, int BlockM, int BlockN, bool PairRows = false>
Fp8A8TmaDescriptors make_fp8_a8_tma_descriptors(const std::uint8_t* activation_codes,
                                                const std::uint8_t* weight_codes,
                                                std::int32_t tokens) {
    constexpr std::uint32_t kRowBytes = 64;
    // A paired block computes both branches, so its weight box is one branch tall and the
    // producer issues one request per branch.
    constexpr std::uint32_t kWeightBoxRows = PairRows ? BlockN / 2 : BlockN;"""),

    ("""        Geometry::kInputRows, kRowBytes, BlockN, "encode fp8 weight codes TMA");""",
     """        Geometry::kInputRows, kRowBytes, kWeightBoxRows, "encode fp8 weight codes TMA");"""),

    # 2. параметры ядра
    ("""template <class Geometry, class Schedule, class Epilogue, class Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_tma_kernel(
    const __grid_constant__ Fp8A8TmaDescriptors descriptors,
    const float* __restrict__ activation_scales, const __nv_bfloat16* __restrict__ weight_scales,
    const __grid_constant__ Epilogue epilogue, const __grid_constant__ Output output) {
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared          = *reinterpret_cast<Fp8A8TmaSharedStorage<Schedule>*>(shared_bytes);
    const int token_begin = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin   = static_cast<int>(blockIdx.x) * Schedule::kBlockN;""",
     """template <class Geometry, class Schedule, class Epilogue, class Output,
          class RowPolicy = Fp8MmaIdentityRows, bool PairRows = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_tma_kernel(
    const __grid_constant__ Fp8A8TmaDescriptors descriptors,
    const float* __restrict__ activation_scales, const __nv_bfloat16* __restrict__ weight_scales,
    const __grid_constant__ Epilogue epilogue, const __grid_constant__ Output output,
    const __grid_constant__ RowPolicy row_policy) {
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);
    static_assert(!PairRows || (Schedule::kBlockN % 2) == 0);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared              = *reinterpret_cast<Fp8A8TmaSharedStorage<Schedule>*>(shared_bytes);
    constexpr int kBranchRows = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    const int token_begin     = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin       = static_cast<int>(blockIdx.x) * kBranchRows;"""),

    # 3. продюсер: два запроса на стадию, когда ветвей две
    ("""                fp8_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                k_tile * Schedule::kRowBytes, row_begin, &shared.full[stage]);""",
     """                if constexpr (PairRows) {
                    fp8_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                    k_tile * Schedule::kRowBytes,
                                    row_policy.weight_row(row_begin, 0), &shared.full[stage]);
                    fp8_tma_load_2d(tensors.b_codes[stage] + kBranchRows * Schedule::kRowBytes,
                                    &descriptors.b_codes, k_tile * Schedule::kRowBytes,
                                    row_policy.weight_row(row_begin, kBranchRows),
                                    &shared.full[stage]);
                } else {
                    fp8_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                    k_tile * Schedule::kRowBytes, row_begin, &shared.full[stage]);
                }"""),

    # 4. эпилог: масштаб веса берётся по политике строк
    ("""            const int parent_row0 = row_begin + local_row0;
            const int parent_row1 = parent_row0 + 1;""",
     """            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1);"""),

    # 5. финальная запись: пара векторов, когда ветвей две
    ("""    constexpr int kVectorsPerRow = Schedule::kBlockN / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int local_token = task / kVectorsPerRow;
        const int row_vector  = task - local_token * kVectorsPerRow;
        const uint4 values =
            load_vec<uint4>(shared_output + local_token * kOutputStride + row_vector * 8);
        output.store_vector(row_begin + row_vector * 8, token_begin + local_token, values);
    }""",
     """    constexpr int kVectorsPerRow = kBranchRows / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int local_token = task / kVectorsPerRow;
        const int row_vector  = task - local_token * kVectorsPerRow;
        const auto* row_base  = shared_output + local_token * kOutputStride + row_vector * 8;
        const uint4 values    = load_vec<uint4>(row_base);
        if constexpr (PairRows) {
            output.store_pair_vector(row_begin + row_vector * 8, token_begin + local_token, values,
                                     load_vec<uint4>(row_base + kBranchRows));
        } else {
            output.store_vector(row_begin + row_vector * 8, token_begin + local_token, values);
        }
    }"""),

    # 6. общая функция запуска
    ("""template <class Geometry, class Schedule, class Epilogue, class Output>
void fp8_a8_tma_launch(const std::uint8_t* activation_codes, const float* activation_scales,
                       const std::uint8_t* weight_codes, const __nv_bfloat16* weight_scales,
                       std::int32_t tokens, Epilogue epilogue, Output output, cudaStream_t stream) {
    constexpr std::size_t kSharedBytes = sizeof(Fp8A8TmaSharedStorage<Schedule>);
    static_assert(kSharedBytes <= 99 * 1024);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
    if (attribute != cudaSuccess) { throw std::runtime_error("fp8 TMA shared memory attribute"); }

    const Fp8A8TmaDescriptors descriptors =
        make_fp8_a8_tma_descriptors<Geometry, Schedule::kBlockM, Schedule::kBlockN>(
            activation_codes, weight_codes, tokens);
    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, activation_scales,
                                                               weight_scales, epilogue, output);
}""",
     """template <class Geometry, class Schedule, class Epilogue, class Output,
          class RowPolicy = Fp8MmaIdentityRows, bool PairRows = false>
void fp8_a8_tma_launch(const std::uint8_t* activation_codes, const float* activation_scales,
                       const std::uint8_t* weight_codes, const __nv_bfloat16* weight_scales,
                       std::int32_t tokens, Epilogue epilogue, Output output, cudaStream_t stream,
                       RowPolicy row_policy = {}) {
    constexpr std::size_t kSharedBytes = sizeof(Fp8A8TmaSharedStorage<Schedule>);
    static_assert(kSharedBytes <= 99 * 1024);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
    if (attribute != cudaSuccess) { throw std::runtime_error("fp8 TMA shared memory attribute"); }

    const Fp8A8TmaDescriptors descriptors =
        make_fp8_a8_tma_descriptors<Geometry, Schedule::kBlockM, Schedule::kBlockN, PairRows>(
            activation_codes, weight_codes, tokens);
    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(
            descriptors, activation_scales, weight_scales, epilogue, output, row_policy);
}"""),
])

# --- MLP gate/up -------------------------------------------------------------------------------
patch('src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu', [
    ("""#include "ops/linear/fp8/fp8_config.h\"""",
     """#include "ops/linear/fp8/fp8_a8_tma.cuh"
#include "ops/linear/fp8/fp8_config.h\""""),

    ("""    launch_fp8_a8_quantize(x, weight, scratch, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, out, scratch, x.ne[1], stream);""",
     """    launch_fp8_a8_quantize(x, weight, scratch, stream);
    using TmaSchedule = typename Fp8LinearA8TmaSchedule<Geometry>::Type;
    using TmaRows     = Fp8SwiGluRows<TmaSchedule::kBlockN / 2, kIntermediate>;
    if (fp8_a8_tma_applies<Geometry, TmaSchedule>(x.ne[1])) {
        fp8_a8_tma_launch<Geometry, TmaSchedule, Fp8IdentityEpilogue, Fp8SwiGluOutput, TmaRows,
                          true>(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), x.ne[1], Fp8IdentityEpilogue{},
            Fp8SwiGluOutput{static_cast<__nv_bfloat16*>(out.data), kIntermediate}, stream,
            TmaRows{});
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, out, scratch, x.ne[1], stream);"""),
])
print('парная запись строк заведена, linear_swiglu на TMA')

# --- плоский linear переводится на ту же общую функцию запуска ----------------------------------
patch('src/ops/linear/fp8/fp8_a8.cu', [
    ("""// The TMA route needs a whole token tile: partial tiles would have the epilogue write rows that
// do not exist. Everything else keeps the cp.async route.
template <class Geometry, class Schedule>
void launch_tma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    constexpr std::size_t kSharedBytes = sizeof(Fp8A8TmaSharedStorage<Schedule>);
    static_assert(kSharedBytes <= 99 * 1024);
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    const Fp8A8TmaDescriptors descriptors =
        make_fp8_a8_tma_descriptors<Geometry, Schedule::kBlockM, Schedule::kBlockN>(
            workspace.codes, static_cast<const std::uint8_t*>(weight.qdata), tokens);

    static const cudaError_t attribute = cudaFuncSetAttribute(
        fp8_a8_tma_kernel<Geometry, Schedule, Fp8IdentityEpilogue, Fp8ContiguousOutput>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
    CUDA_CHECK(attribute);

    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    fp8_a8_tma_kernel<Geometry, Schedule, Fp8IdentityEpilogue, Fp8ContiguousOutput>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(
            descriptors, workspace.scales, static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}""",
     """// The TMA route needs a whole token tile: partial tiles would have the epilogue write rows that
// do not exist. Everything else keeps the cp.async route.
template <class Geometry, class Schedule>
void launch_tma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    fp8_a8_tma_launch<Geometry, Schedule>(
        workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows}, stream);
    CUDA_CHECK(cudaGetLastError());
}"""),
])
print('плоский linear переведён на общую функцию')
