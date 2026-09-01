#!/usr/bin/env python3
"""Второй круг абляций routed MoE-GEMM: круг первый показал, что ни MMA (6.5%), ни распаковка
Q4 (0.8%) не держат ядро. Значит держит подвоз, обмен через shared или эпилог — по одному.

  bs     -- активации стейджатся нулём байт: инструкция cp.async остаётся, глобального чтения нет
  cr     -- то же для кодов веса
  epi    -- эпилог не пишет в глобальную память
  ldm    -- ldmatrix заменён на присваивание адреса: чтение из shared исчезает, адресация цела
  st3    -- КАНДИДАТ, не абляция: три стадии конвейера вместо двух

Кроме st3 результаты неверны намеренно.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")


def sub(old, new, what, expect=1):
    global t
    if t.count(old) != expect:
        raise SystemExit("anchor %s count %d (expected %d)" % (what, t.count(old), expect))
    t = t.replace(old, new)


if MODE == "bs":
    sub("""                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);""",
        """                // ABLATION: активации не читаются из глобальной памяти; инструкция и
                // адресация остаются, поэтому измеряется именно трафик подвоза.
                cp_async_zfill<16, Cache::cg>(dst, src, 0);""",
        "bs", expect=2)

elif MODE == "cr":
    sub("""                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);""",
        """                // ABLATION: коды веса не читаются из глобальной памяти.
                cp_async_zfill<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                              &codes[gi * 32 + half * 16], 0);""",
        "cr")

elif MODE == "epi":
    sub("""                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][0]) * acc[mi + 2][ni][0]);
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][2]) * acc[mi + 2][ni][2]);
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][1]) * acc[mi + 2][ni][1]);
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][3]) * acc[mi + 2][ni][3]);
                    }""",
        """                    // ABLATION: SiLU-гейт считается, запись в глобальную память снята
                    // условием, ложность которого компилятор доказать не может.
                    const float g0 = silu(acc[mi][ni][0]) * acc[mi + 2][ni][0];
                    const float g1 = silu(acc[mi][ni][1]) * acc[mi + 2][ni][1];
                    const float g2 = silu(acc[mi][ni][2]) * acc[mi + 2][ni][2];
                    const float g3 = silu(acc[mi][ni][3]) * acc[mi + 2][ni][3];
                    if (g0 != g0 && g1 != g1 && g2 != g2 && g3 != g3) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(g0);
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(g3);
                    }""",
        "epi")

elif MODE == "ldm":
    sub("""                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));""",
        """                        // ABLATION: чтение операнда A из shared снято, адресация цела.
                        const unsigned aa =
                            smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]);
                        af[mi][0] = aa;
                        af[mi][1] = aa ^ 1u;
                        af[mi][2] = aa ^ 2u;
                        af[mi][3] = aa ^ 3u;""",
        "ldm")
    sub("""                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));""",
        """                        const unsigned bb =
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]);
                        bf[ni][0] = bb;
                        bf[ni][1] = bb ^ 1u;""",
        "ldm_b")

elif MODE == "st3":
    sub("constexpr int kExpertStages            = 2;",
        "constexpr int kExpertStages            = 3;",
        "stages")

else:
    raise SystemExit("mode must be bs|cr|epi|ldm|st3")

F.write_text(t, encoding="utf-8")
print("patched:", MODE)
