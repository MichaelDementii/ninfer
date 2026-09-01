#!/usr/bin/env python3
"""П4, часть 4: неполный токенный тайл.

Маршрут TMA сейчас требует, чтобы число токенов было кратно 256. У реального промпта хвостовой
чанк почти всегда некратен, и он целиком уходит на прежний путь. TMA сам заполняет нулями всё, что
выходит за границу дескриптора, поэтому арифметика для несуществующих строк безопасна — достаточно
не записывать их и не читать для них масштаб активации.

Ограничение остаётся одно: тайл в 256 токенов должен помещаться хотя бы раз, то есть промпты
короче 256 токенов по-прежнему идут прежним путём.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

patch('src/ops/linear/fp8/fp8_a8_tma.cuh', [
    # 1. ядро принимает число токенов и укорачивает эпилог
    ("""    const __grid_constant__ Epilogue epilogue, const __grid_constant__ Output output,
    const __grid_constant__ RowPolicy row_policy) {""",
     """    std::int32_t tokens, const __grid_constant__ Epilogue epilogue,
    const __grid_constant__ Output output, const __grid_constant__ RowPolicy row_policy) {"""),

    # 2. масштаб активации только для существующих строк
    ("""        const float activation_scale0 = activation_scales[token0];
        const float activation_scale1 = activation_scales[token1];""",
     """        // TMA zero-fills the rows past the end, so their accumulators are harmless; they are
        // simply never stored. The scale is still read only for rows that exist.
        const float activation_scale0 = token0 < tokens ? activation_scales[token0] : 0.0F;
        const float activation_scale1 = token1 < tokens ? activation_scales[token1] : 0.0F;"""),

    # 3. запись только существующих строк
    ("""    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int local_token = task / kVectorsPerRow;
        const int row_vector  = task - local_token * kVectorsPerRow;
        const auto* row_base  = shared_output + local_token * kOutputStride + row_vector * 8;
        const uint4 values    = load_vec<uint4>(row_base);""",
     """    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int local_token = task / kVectorsPerRow;
        if (token_begin + local_token >= tokens) { continue; }
        const int row_vector = task - local_token * kVectorsPerRow;
        const auto* row_base = shared_output + local_token * kOutputStride + row_vector * 8;
        const uint4 values   = load_vec<uint4>(row_base);"""),

    # 4. запуск: сетка по потолку, число токенов в ядро
    ("""    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN, tokens / Schedule::kBlockM);
    fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(
            descriptors, activation_scales, weight_scales, epilogue, output, row_policy);""",
     """    const int token_tiles = (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM;
    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN, token_tiles);
    fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(
            descriptors, activation_scales, weight_scales, tokens, epilogue, output, row_policy);"""),

])

# --- условие маршрута: достаточно одного полного тайла, хвост допускается ------------------------
patch('src/ops/linear/fp8/fp8_a8_schedule.cuh', [
    ("""// Guard for the TMA route: a whole token tile, and both extents divisible by the block tile.""",
     """// Guard for the TMA route: at least one token tile, and both weight extents divisible by the
// block tile. A partial trailing tile is allowed; TMA zero-fills it and the epilogue skips it."""),

    ("""    if (tokens < Schedule::kBlockM || (tokens % Schedule::kBlockM) != 0) { return false; }""",
     """    if (tokens < Schedule::kBlockM) { return false; }"""),
])
print('неполный тайл поддержан')
