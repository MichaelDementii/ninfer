#!/usr/bin/env python3
"""Dva dolevyh pribora po vnimaniyu. Ni odin ne zamenyaet rabotu -- oba ee sokrashchayut,
poetomu naklon chitaetsya pryamo, bez popravki na stoimost zameny.

  exp -- polovina eksponent v softmakse: p01 i p11 beryutsya iz p00 i p10.
         Zapisi v p_s te zhe, chetyre na nt; schitaetsya dva exp2 vmesto chetyreh.
  vld -- polovina chteniy V iz obshchey pamyati: nechetnoe n pereispolzuet fragment
         chetnogo. Chislo mma to zhe, chislo ldmatrix vdvoe menshe.

Rezultat neveren namerenno.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
t = F.read_text(encoding="utf-8")


def sub(old, new, what):
    global t
    if t.count(old) != 1:
        raise SystemExit("anchor %s count %d" % (what, t.count(old)))
    t = t.replace(old, new)


if MODE == "exp":
    sub("""                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;""",
        """                // ABLIACIYA: eksponenta ne schitaetsya, beretsya sosednyaya.
                const float p01 = score[nt][1] > -CUDART_INF_F ? p00 : 0.0f;""",
        "p01")
    sub("""                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;""",
        """                // ABLIACIYA: eksponenta ne schitaetsya, beretsya sosednyaya.
                const float p11 = score[nt][3] > -CUDART_INF_F ? p10 : 0.0f;""",
        "p11")
elif MODE == "vld":
    # vf_prev ob'yavlyaetsya SNARUZHI cikla i imenno kak obychnaya lokalnaya peremennaya:
    # static vnutri device-funkcii ushel by v globalnuyu pamyat i sdelal by zamer bessmyslennym.
    sub("""#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));""",
        """            // ABLIACIYA: nechetnoe n pereispolzuet fragment chetnogo -- chteniy vdvoe
            // menshe pri tom zhe chisle mma.
            unsigned vf_prev[2] = {0u, 0u};
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                if ((n & 1) == 0) {
                    ldmatrix_x2_t(vf[0], vf[1],
                                  smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                    vf_prev[0] = vf[0];
                    vf_prev[1] = vf[1];
                } else {
                    vf[0] = vf_prev[0];
                    vf[1] = vf_prev[1];
                }""",
        "vld")
else:
    raise SystemExit("mode must be exp|vld")

F.write_text(t, encoding="utf-8")
print("abliaciya vnimaniya:", MODE)
