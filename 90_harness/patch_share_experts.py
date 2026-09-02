#!/usr/bin/env python3
"""Verhnyaya granica priza dedupliкacii ekspertov v dekode.

Yadra d3 i d4 chitayut ekspertov po tokenu: token_ids[token * kTopK + ...]. Kogda linii
raznye, eksperty raznye, i vesa chitayutsya zanovo dlya kazhdogo tokena -- imenno poetomu
moe_d4_token ne amortiziruetsya po liniyam (x2.56 vremeni za x2.61 raboty).

Ablyaciya: vse tokeny berut ekspertov PERVOGO tokena. Rabota ta zhe (te zhe puski, te zhe
plitki, te zhe instrukcii), a razlichnyh ekspertov na raund -- vosem vmesto sta s lishnim.
Rezultat neveren namerenno. Razryv i est verhnyaya granica togo, chto mozhet dat lyubaya
gruppirovka po ekspertam.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
F = ROOT / "src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu"
t = F.read_text(encoding="utf-8")

SUBS = [
    ("            const int expert   = token_ids[token * kTopK + path];",
     "            // ABLIACIYA: vse tokeny berut ekspertov pervogo tokena.\n"
     "            const int expert   = token_ids[0 * kTopK + path];"),
    ("            const int expert = token_ids[token * kTopK + warp];",
     "            // ABLIACIYA: vse tokeny berut ekspertov pervogo tokena.\n"
     "            const int expert = token_ids[0 * kTopK + warp];"),
]
for old, new in SUBS:
    if t.count(old) != 1:
        raise SystemExit("anchor %r count %d" % (old[:50], t.count(old)))
    t = t.replace(old, new)

F.write_text(t, encoding="utf-8")
print("vse tokeny delyat ekspertov pervogo")
