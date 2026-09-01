#!/usr/bin/env python3
"""Диагностика префилльного внимания на fp8-KV: баланс варпов между QK и PV.

Ядро расщеплено: варпы 0..7 считают QK, варпы 8..15 — PV. Число QK-варпов жёстко задано как
`2 * RowTiles`, а деление колонок — как `warp & 1`. Здесь оно вынесено в параметр `QKSplits`,
чтобы можно было замерить и 8/8 (как сейчас), и 4/12.

Модель бенча (QK по 419 TFLOP/s, PV по 209.5) предсказывает, что время равно
`max(16/p · t_QK, 16/(16-p) · t_PV)`: при p=8 это 16726 мкс против измеренных 16151.5, а при p=4 —
столько же. Другая модель, с измеренными потолками, даёт при p=4 на 18% меньше. Модели расходятся,
поэтому это замер, а не правка.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

F = 'src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh'

patch(F, [
    # 1. число QK-варпов как параметр
    ("""inline constexpr int kCausalPromptFp8ProducerWarps = 2 * kCausalPromptFp8RowTiles;""",
     """// How many warps share one row tile on the QK side. The remaining warps do PV, so this sets
// the balance between the two halves of the pipeline.
inline constexpr int kCausalPromptFp8QKSplits      = 2;
inline constexpr int kCausalPromptFp8ProducerWarps =
    kCausalPromptFp8QKSplits * kCausalPromptFp8RowTiles;"""),

    ("""    constexpr int QKNt          = (Bc / 2) / 8;""",
     """    constexpr int QKSplits      = kCausalPromptFp8QKSplits;
    constexpr int QKNt          = (Bc / QKSplits) / 8;"""),

    # 2. отображение варпа на строки и колонки
    ("""        const int row0 = (warp >> 1) * 16 + gid;""",
     """        const int row0 = (warp / QKSplits) * 16 + gid;"""),

    ("""            const int row_base = (warp >> 1) * 16;
            const int col_half = warp & 1;
            const int col_base = col_half * (Bc / 2);""",
     """            const int row_base = (warp / QKSplits) * 16;
            const int col_half = warp % QKSplits;
            const int col_base = col_half * (Bc / QKSplits);"""),

    # 3. сведение частичных максимумов и сумм по числу частей
    ("""            asm volatile("bar.sync 1, 256;" ::: "memory");

            bm0                     = fmaxf(partial_m_s[row0], partial_m_s[Br + row0]);
            bm1                     = fmaxf(partial_m_s[row1], partial_m_s[Br + row1]);""",
     """            asm volatile("bar.sync 1, %0;" ::"r"(ProducerWarps * 32) : "memory");

            bm0 = partial_m_s[row0];
            bm1 = partial_m_s[row1];
#pragma unroll
            for (int part = 1; part < QKSplits; ++part) {
                bm0 = fmaxf(bm0, partial_m_s[part * Br + row0]);
                bm1 = fmaxf(bm1, partial_m_s[part * Br + row1]);
            }"""),

    ("""            asm volatile("bar.sync 1, 256;" ::: "memory");
            if (col_half == 0 && lid == 0) {
                const float tile_l0 = partial_l_s[row0] + partial_l_s[Br + row0];
                const float tile_l1 = partial_l_s[row1] + partial_l_s[Br + row1];""",
     """            asm volatile("bar.sync 1, %0;" ::"r"(ProducerWarps * 32) : "memory");
            if (col_half == 0 && lid == 0) {
                float tile_l0 = partial_l_s[row0];
                float tile_l1 = partial_l_s[row1];
#pragma unroll
                for (int part = 1; part < QKSplits; ++part) {
                    tile_l0 += partial_l_s[part * Br + row0];
                    tile_l1 += partial_l_s[part * Br + row1];
                }"""),
])
print('баланс варпов вынесен в параметр')
