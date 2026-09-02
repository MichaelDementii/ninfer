#!/usr/bin/env python3
"""Kandidat 18: porog shirokogo plana zadan v tokenah, a reshaet chislo kolonok na eksperta.

wide_plan = tokens >= kSparseMoePrefillWideMin = 768 vybiraet plitku kolonok shirinoy 64.
Kolonok na eksperta v srednem tokens * kTopK / kExperts = tokens / 32 pri 8 iz 256.
Pri 768 tokenah eto 24 kolonki na plitku v 64 -- 62% plitki schitaet pustotu.
Rovnaya plitka trebuet tokens >= 64 * 256 / 8 = 2048.

Argument -- novoe znachenie poroga.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
VALUE = int(sys.argv[2])
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill.h"
t = F.read_text(encoding="utf-8")
OLD = "inline constexpr std::int32_t kSparseMoePrefillWideMin      = 768;"
NEW = "inline constexpr std::int32_t kSparseMoePrefillWideMin      = %d;" % VALUE
if t.count(OLD) != 1:
    raise SystemExit("anchor count %d" % t.count(OLD))
F.write_text(t.replace(OLD, NEW), encoding="utf-8")
print("kSparseMoePrefillWideMin = %d" % VALUE)
