#!/usr/bin/env python3
"""ИСПРАВЛЕНИЕ ПРИБОРА. Первый круг заменял mma.sync на «дешёвую зависимость» из десяти
инструкций (пять XOR, AND, четыре FADD). Замена оказалась дороже заменяемого, поэтому те 6.5%
— нижняя граница доли MMA, а не её значение.

Здесь прибор другой: число тензорных инструкций уменьшается кратно, а весь остальной код —
стейджинг, распаковка, ldmatrix, свёртка масштабов, эпилог — остаётся ровно тем же. Доля MMA
восстанавливается по наклону.

  half     -- mma только для mi < 2 из 4: половина тензорных инструкций
  quarter  -- mma только для mi < 1 из 4: четверть
  none     -- mma не выполняется вовсе; операнды остаются живыми одним XOR на инструкцию
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")

OLD = """                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }"""

BODY = """                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);"""

if MODE == "half":
    new = ("                        // ПРИБОР: половина тензорных инструкций, остальное без изменений.\n"
           "                        for (int mi = 0; mi < 2; ++mi) {\n" + BODY + "\n                        }")
elif MODE == "quarter":
    new = ("                        // ПРИБОР: четверть тензорных инструкций.\n"
           "                        for (int mi = 0; mi < 1; ++mi) {\n" + BODY + "\n                        }")
elif MODE == "none":
    new = """                        // ПРИБОР: тензорной инструкции нет; операнды удерживаются живыми
                        // одним XOR на каждую снятую инструкцию, что дешевле её самой.
                        for (int mi = 0; mi < 4; ++mi) {
                            live ^= af[mi][0] ^ bf[ni][0];
                        }"""
else:
    raise SystemExit("mode must be half|quarter|none")

if t.count(OLD) != 1:
    raise SystemExit("anchor count %d" % t.count(OLD))
t = t.replace(OLD, new)

if MODE == "none":
    decl_old = """        float acc[4][WarpNT][4] = {};"""
    decl_new = """        float acc[4][WarpNT][4] = {};
        unsigned live           = 0u;"""
    if t.count(decl_old) != 1:
        raise SystemExit("acc decl count %d" % t.count(decl_old))
    t = t.replace(decl_old, decl_new)
    guard_old = """        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;"""
    guard_new = """        if (live == 0xdeadbeefu) { acc[0][0][0] += 1.0F; }
        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;"""
    if t.count(guard_old) != 1:
        raise SystemExit("epilogue guard count %d" % t.count(guard_old))
    t = t.replace(guard_old, guard_new)

F.write_text(t, encoding="utf-8")
print("mma fraction patched:", MODE)
