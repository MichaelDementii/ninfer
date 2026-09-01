#!/usr/bin/env python3
"""nvfp4: масштабы активаций блоками, чтобы TMA везла их крупными запросами.

Дескриптор `a_scales` описывает коробку **16 байт при шаге строки kGroupsPerRow** (1088 для
K=17408). На стадию это 256 отдельных крошечных запросов ради 4096 байт полезной нагрузки.
Замерено на стенде (`90_harness/tmaparts.cu`, `tmablock.cu`): добавление одного этого дескриптора
к подвозу кодов даёт +16.7% байт и **+73% времени**; та же нагрузка блоком идёт за 356.0 мкс
вместо 527.4 и возвращает скорость к 7009 ГБ/с — ровно как у одних кодов.

Ядро на 81% упирается в подвоз: продюсер без счёта занимает 575.9 мкс из 712.7.

Раскладка сходится без перестановки: тайл `(токенный тайл, пара k-тайлов)` — это те же 4096 байт
подряд в порядке `[токен][слово]`, то есть ровно тот образ, который TMA и так кладёт в shared.
Меняются адрес записи в квантователе, форма дескриптора и координата запроса.

Блочная раскладка пишется **только под маршрут TMA**: тот же буфер читают `nvfp4_small_t_kernel`
и `nvfp4_gemv_kernel`. Общий дескриптор используют четыре маршрута — `linear`, `linear_add`,
`attn_input_proj`, `gdn_input_proj`, и предикат маршрута у всех четырёх один и тот же, поэтому
флаг ставится единообразно. `linear_swiglu` строит свои дескрипторы отдельно и остаётся на прежней
раскладке.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

# --- 1. предикат маршрута в общем заголовке ------------------------------------------------------
patch('src/ops/linear/nvfp4/nvfp4_w4a4_plan.h', [
    ("""void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream);""",
     """// The TMA route reads activation scales one [256 tokens, 16 groups] tile per request, and wants
// that tile contiguous. Every route that shares make_nvfp4_w4a4_tma_descriptors asks the same
// question, so the predicate lives here.
inline constexpr std::int32_t kNvfp4TmaBlockM = 256;
[[nodiscard]] inline bool nvfp4_w4a4_tma_route(std::int32_t tokens) {
    return tokens >= 1024 && (tokens % kNvfp4TmaBlockM) == 0;
}

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                bool blocked_scales, cudaStream_t stream);"""),
])

# --- 2. квантователь: адрес записи масштаба ------------------------------------------------------
patch('src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh', [
    ("""template <class Geometry, int Threads = 256>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_w4a4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, float input_scale_divisor) {
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);""",
     """// The TMA route reads a whole [BlockM tokens, 16 groups] tile of activation scales per request.
// Row-major by token that tile is 16 bytes per row with the scale plane's row stride, so one stage
// costs BlockM tiny requests. Writing the same bytes tile-contiguous makes the request wide. The
// byte order inside a tile is unchanged, so the shared image the consumer reads is identical.
template <class Geometry, int BlockM>
__device__ __forceinline__ std::int64_t nvfp4_blocked_scale_offset(int token, int group) {
    constexpr int kGroupsPerRow  = Geometry::kInputRows / 16;
    constexpr int kGroupsPerTile = 16;
    constexpr int kTilesPerPlane = kGroupsPerRow / kGroupsPerTile;
    const int token_tile         = token / BlockM;
    const int group_tile         = group / kGroupsPerTile;
    const int tile               = token_tile * kTilesPerPlane + group_tile;
    return static_cast<std::int64_t>(tile) * BlockM * kGroupsPerTile +
           static_cast<std::int64_t>(token - token_tile * BlockM) * kGroupsPerTile +
           (group - group_tile * kGroupsPerTile);
}

template <class Geometry, int Threads = 256, bool BlockedScales = false, int BlockM = 256>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_w4a4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, float input_scale_divisor) {
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);
    static_assert(!BlockedScales || (Geometry::kInputRows / 16) % 16 == 0);"""),

    ("""    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    scales[static_cast<std::int64_t>(token) * kGroupsPerRow + group] = quantized.scale;""",
     """    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    if constexpr (BlockedScales) {
        scales[nvfp4_blocked_scale_offset<Geometry, BlockM>(token, group)] = quantized.scale;
    } else {
        scales[static_cast<std::int64_t>(token) * kGroupsPerRow + group] = quantized.scale;
    }"""),
])

# --- 3. дескриптор и координата запроса ----------------------------------------------------------
patch('src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh', [
    ("""    constexpr std::uint32_t kCodeColumns = 64;
    // TMA's innermost box is at least one 16-byte transaction. A K128 tile consumes the
    // first eight bytes of each row; the second half is harmless look-ahead.
    constexpr std::uint32_t kScaleColumns = 16;""",
     """    constexpr std::uint32_t kCodeColumns = 64;
    // Activation scales arrive tile-contiguous: one [BlockM tokens, 16 groups] tile is BlockM
    // bytes wide and 16 rows tall, so the request is wide instead of BlockM separate 16-byte ones.
    // A K128 tile consumes the first eight of the sixteen group bytes; the rest is look-ahead.
    constexpr std::uint32_t kScaleTileGroups = 16;
    constexpr std::uint64_t kScaleTilesPerPlane =
        static_cast<std::uint64_t>(Geometry::kGroupsPerRow) / kScaleTileGroups;"""),

    ("""    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns, BlockM,
        CU_TENSOR_MAP_SWIZZLE_NONE, "encode activation scales TMA");""",
     """    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, BlockM,
        (static_cast<std::uint64_t>(tokens) / BlockM) * kScaleTilesPerPlane * kScaleTileGroups,
        BlockM, BlockM, kScaleTileGroups, CU_TENSOR_MAP_SWIZZLE_NONE,
        "encode activation scales TMA");"""),

    ("""                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, (k_tile / 2) * 16,
                                  token_begin, &shared.full[stage]);""",
     """                constexpr int kScaleTilesPerPlane = Geometry::kGroupsPerRow / 16;
                const int scale_tile =
                    (token_begin / Schedule::kBlockM) * kScaleTilesPerPlane + k_tile / 2;
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, 0,
                                  scale_tile * 16, &shared.full[stage]);"""),
])

# --- 4. диспетчер линейной формы ------------------------------------------------------------------
patch('src/ops/linear/nvfp4/nvfp4_w4a4.cu', [
    ("""template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_kernel<ActivationGeometry>
        <<<(tasks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}""",
     """template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           bool blocked_scales, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    const int blocks          = (tasks + kThreads - 1) / kThreads;
    if (blocked_scales) {
        nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads, true, kNvfp4TmaBlockM>
            <<<blocks, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              workspace.codes, workspace.scales, tokens,
                                              weight.input_scale_divisor);
    } else {
        nvfp4_w4a4_quantize_kernel<ActivationGeometry>
            <<<blocks, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              workspace.codes, workspace.scales, tokens,
                                              weight.input_scale_divisor);
    }
    CUDA_CHECK(cudaGetLastError());
}"""),

    ("""    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, stream);
        return;""",
     """    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, blocked_scales,
                                                           stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, blocked_scales,
                                                           stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, blocked_scales,
                                                            stream);
        return;"""),

    ("""void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream) {""",
     """void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                bool blocked_scales, cudaStream_t stream) {"""),

    ("""    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {""",
     """    launch_nvfp4_w4a4_quantize(x, weight, workspace, nvfp4_w4a4_tma_route(x.ne[1]), stream);
    const std::int32_t tokens = x.ne[1];
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {"""),
])

# --- 5. остальные точки вызова -------------------------------------------------------------------
for path in ('src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu',
             'src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu',
             'src/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu'):
    patch(path, [
        ("""    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);""",
         """    launch_nvfp4_w4a4_quantize(x, weight, workspace, nvfp4_w4a4_tma_route(x.ne[1]), stream);"""),
    ])

# linear_swiglu строит свои дескрипторы и остаётся на прежней раскладке
patch('src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu', [
    ("""    launch_nvfp4_w4a4_quantize(x, weight, scratch, stream);""",
     """    // This route builds its own descriptors and keeps the row-major scale plane.
    launch_nvfp4_w4a4_quantize(x, weight, scratch, false, stream);"""),
])
print('блочная раскладка масштабов заведена')

# фьюзед-свиглу тоже строит свои дескрипторы: прежняя раскладка
patch('src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp', [
    ("""        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, stream);""",
     """        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        // The fused SwiGLU route builds its own descriptors and keeps the row-major scale plane.
        launch_nvfp4_w4a4_quantize(x, weight, scratch, false, stream);"""),
])
print('фьюзед-свиглу тоже поправлен')

# --- 6. слитый свиглу: тот же приём в его собственных дескрипторах -------------------------------
patch('src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu', [
    ("""    constexpr std::uint32_t kCodeColumns  = 64;
    constexpr std::uint32_t kScaleColumns = 16;""",
     """    constexpr std::uint32_t kCodeColumns      = 64;
    // Activation scales arrive tile-contiguous, one [BlockM tokens, 16 groups] tile per request.
    constexpr std::uint32_t kScaleTileGroups = 16;
    constexpr std::uint64_t kScaleTilesPerPlane =
        static_cast<std::uint64_t>(Geometry::kGroupsPerRow) / kScaleTileGroups;"""),

    ("""    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns, Schedule::kBlockM,
        CU_TENSOR_MAP_SWIZZLE_NONE, "encode LinearSwiGLU activation scales TMA");""",
     """    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Schedule::kBlockM,
        (static_cast<std::uint64_t>(tokens) / Schedule::kBlockM) * kScaleTilesPerPlane *
            kScaleTileGroups,
        Schedule::kBlockM, Schedule::kBlockM, kScaleTileGroups, CU_TENSOR_MAP_SWIZZLE_NONE,
        "encode LinearSwiGLU activation scales TMA");"""),
])

patch('src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh', [
    ("""                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, (k_tile / 2) * 16,
                                  token_begin, &shared.full[stage]);""",
     """                constexpr int kScaleTilesPerPlane = Geometry::kGroupsPerRow / 16;
                const int scale_tile =
                    (token_begin / Schedule::kBlockM) * kScaleTilesPerPlane + k_tile / 2;
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, 0,
                                  scale_tile * 16, &shared.full[stage]);"""),
])

# теперь и свиглу хочет блочную раскладку
patch('src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu', [
    ("""    // This route builds its own descriptors and keeps the row-major scale plane.
    launch_nvfp4_w4a4_quantize(x, weight, scratch, false, stream);""",
     """    // This route runs the MMA kernel, never TMA, so the scale plane stays row-major.
    launch_nvfp4_w4a4_quantize(x, weight, scratch, false, stream);"""),
])
patch('src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp', [
    ("""        // The fused SwiGLU route builds its own descriptors and keeps the row-major scale plane.
        launch_nvfp4_w4a4_quantize(x, weight, scratch, false, stream);""",
     """        // Inside the TMA case, so this route always reads tile-contiguous scales. Note its own
        // predicate admits every multiple of 256 from 256 up, which is wider than the shared one.
        launch_nvfp4_w4a4_quantize(x, weight, scratch, true, stream);"""),
])
print('слитый свиглу тоже на блочной раскладке')
