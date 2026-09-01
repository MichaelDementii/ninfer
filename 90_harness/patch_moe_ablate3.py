#!/usr/bin/env python3
"""Абляции подвоза в MoE-ядрах префилла. Текст стейджинга одинаков в нескольких маршрутах,
поэтому патчатся все вхождения: трасса разделяет ядра по имени, и один прогон даёт несколько
точек сразу.

  bs   -- активации стейджатся нулём байт: инструкция cp.async и адресация остаются,
          глобального чтения нет
  cr   -- то же для кодов веса
  ldm  -- ldmatrix.x4 операнда A заменён присваиванием адреса: чтения из shared нет
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")

PAIRS = {
    "bs": [(
        "cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);",
        "cp_async_zfill<16, Cache::cg>(dst, src, 0);  // ABLATION: без глобального чтения",
    )],
    "cr": [(
        """cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);""",
        """// ABLATION: коды веса не читаются из глобальной памяти.
                cp_async_zfill<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                              &codes[gi * 32 + half * 16], 0);""",
    )],
    "ldm": [(
        """ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));""",
        """// ABLATION: чтение операнда A из shared снято, адресация цела.
                        const unsigned aa =
                            smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]);
                        af[mi][0] = aa;
                        af[mi][1] = aa ^ 1u;
                        af[mi][2] = aa ^ 2u;
                        af[mi][3] = aa ^ 3u;""",
    )],
}

if MODE not in PAIRS:
    raise SystemExit("mode must be one of %s" % ", ".join(PAIRS))

total = 0
for old, new in PAIRS[MODE]:
    n = t.count(old)
    if n == 0:
        raise SystemExit("anchor for %s not found" % MODE)
    t = t.replace(old, new)
    total += n

F.write_text(t, encoding="utf-8")
print("patched %s in %d places" % (MODE, total))
