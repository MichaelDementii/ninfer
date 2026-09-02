#!/usr/bin/env python3
"""Abliaciya raspakovki V vo vnimanii -- DOLEY, a ne zamenoy.

Urok proshlogo kruga: esli zamena stoit bolshe instrukciy, chem to, chto ona zamenyaet,
rezultat -- tolko nizhnyaya granica. Poetomu zdes chislo preobrazovaniy sokrashchaetsya
vdvoe (dve pary vmesto chetyreh), a zagruzka i zapis ostayutsya bayt v bayt temi zhe.
Rezultat neveren namerenno; meryaem naklon, a ne otvet.

Yadro ustroeno kak konveyer: otdelnye varpy raspakovyvayut V parallelno s MMA. Esli oni
ne na kriticheskom puti, polovina raboty nichego ne izmenit -- i eto tozhe otvet.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
F = ROOT / "src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
t = F.read_text(encoding="utf-8")

OLD = """    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }"""

NEW = """    // ABLIACIYA: polovina preobrazovaniy. Zagruzka i zapis te zhe bayt v bayt,
    // arifmetika raspakovki vdvoe menshe. Rezultat neveren namerenno.
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    packed[2] = packed[0];
    packed[3] = packed[1];"""

if t.count(OLD) != 1:
    raise SystemExit("anchor count %d" % t.count(OLD))
F.write_text(t.replace(OLD, NEW), encoding="utf-8")
print("raspakovka V sokrashchena vdvoe")
