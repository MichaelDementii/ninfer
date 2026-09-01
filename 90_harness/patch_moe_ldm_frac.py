#!/usr/bin/env python3
"""Проверка гипотезы «ядро упирается в ldmatrix, а не в MMA».

На плитке варпа 4x1 (WarpNT = 1) фрагмент A читается ради ОДНОЙ плитки B: за k-тайл варп делает
20 ldmatrix (4 x4 на A и 1 x2 на B, четыре раза по ki) на 16 mma. Инструкций чтения из shared
больше, чем тензорных. Легальный способ это исправить — расширить плитку по N (WarpNT = 2), то
есть ExpertBN = 128; прибор ниже проверяет саму гипотезу, не меняя ничего другого.

  halfa  -- фрагмент A читается для чётных mi и переиспользуется для нечётных: A-загрузок вдвое
            меньше, число mma и всё прочее неизменно
  nob    -- ldmatrix операнда B заменён на переиспользование предыдущего ni

Результаты неверны намеренно.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")

OLD_A = """                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }"""

NEW_A = """                    // ПРИБОР: вдвое меньше чтений операнда A из shared, число mma то же.
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = (mi & ~1) * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        if ((mi & 1) == 0) {
                            ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                        smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                        } else {
                            af[mi][0] = af[mi - 1][0];
                            af[mi][1] = af[mi - 1][1];
                            af[mi][2] = af[mi - 1][2];
                            af[mi][3] = af[mi - 1][3];
                        }
                    }"""

OLD_B = """                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));"""

NEW_B = """                        // ПРИБОР: чтение операнда B из shared снято на всех ni, кроме нулевого.
                        if (ni == 0) {
                            ldmatrix_x2(
                                bf[ni][0], bf[ni][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                        } else {
                            bf[ni][0] = bf[0][0];
                            bf[ni][1] = bf[0][1];
                        }"""

if MODE == "halfa":
    if t.count(OLD_A) == 0:
        raise SystemExit("A anchor not found")
    t = t.replace(OLD_A, NEW_A)
elif MODE == "nob":
    if t.count(OLD_B) == 0:
        raise SystemExit("B anchor not found")
    t = t.replace(OLD_B, NEW_B)
else:
    raise SystemExit("mode must be halfa|nob")

F.write_text(t, encoding="utf-8")
print("ldmatrix instrument:", MODE)
