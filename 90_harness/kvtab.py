#!/usr/bin/env python3
"""Svodka zamera skorosti po kodekam KV: mediana dvuh nog, otnoshenie k bf16 i k int8."""
import re, statistics, sys, collections

rows = collections.defaultdict(list)          # (mtp, kv, pp) -> [(pref_tok_s, dec_tok_s, ms_round)]
kvb  = {}
pat = re.compile(
    r"^\s+(?:L/)?mtp(\d)/(\w+)(?:/(\w))?\s+pp=(\d+)\s+pref_tok_s=\s*([\d.]+)\s+pref_s=\s*([\d.]+)"
    r"\s+dec_tok_s=\s*([\d.]+)\s+ms_round=\s*([\d.]+)\s+acc=(\S*)\s+kv_B=(\d+)")
for line in open("/root/exp/kv_speed.log", encoding="utf-8", errors="replace"):
    m = pat.match(line.rstrip("\n"))
    if not m:
        continue
    mtp, kv, leg, pp, ptok, psec, dtok, msr, acc, kb = m.groups()
    rows[(mtp, kv, int(pp))].append((float(ptok), float(dtok), float(msr)))
    kvb[kv] = int(kb)

def med(vals, i):
    return statistics.median(v[i] for v in vals)

ORDER = ["bf16", "int8", "fp8", "nvfp4", "k8v4"]
for mtp in ("0", "3"):
    pps = sorted({pp for (m_, k_, pp) in rows if m_ == mtp})
    if not pps:
        continue
    metric, idx, better = ("dekod tok/s", 1, +1) if mtp == "0" else ("ms/raund", 2, -1)
    print(f"\n===== mtp={mtp}: {metric} =====")
    print(f"{'kodek':7} " + " ".join(f"{('pp'+str(p)):>26}" for p in pps))
    base = {p: med(rows[(mtp, 'int8', p)], idx) for p in pps if (mtp, 'int8', p) in rows}
    bf   = {p: med(rows[(mtp, 'bf16', p)], idx) for p in pps if (mtp, 'bf16', p) in rows}
    for kv in ORDER:
        cells = []
        for p in pps:
            key = (mtp, kv, p)
            if key not in rows:
                cells.append(f"{'-':>26}"); continue
            v = med(rows[key], idx)
            n = len(rows[key])
            spread = (max(x[idx] for x in rows[key]) - min(x[idx] for x in rows[key])) / v * 100 if n > 1 else 0.0
            di = better * (v - base[p]) / base[p] * 100 if p in base else float('nan')
            db = better * (v - bf[p]) / bf[p] * 100 if p in bf else float('nan')
            cells.append(f"{v:9.2f} i{di:+6.2f}% b{db:+6.2f}%")
        print(f"{kv:7} " + " ".join(cells))

print("\n===== prefill tok/s (mtp=0) =====")
pps = sorted({pp for (m_, k_, pp) in rows if m_ == "0"})
for kv in ORDER:
    cells = []
    for p in pps:
        key = ("0", kv, p)
        if key not in rows:
            cells.append(f"{'-':>26}"); continue
        v = med(rows[key], 0)
        bi = med(rows[("0", "int8", p)], 0)
        bb = med(rows[("0", "bf16", p)], 0)
        cells.append(f"{v:9.1f} i{(v-bi)/bi*100:+6.2f}% b{(v-bb)/bb*100:+6.2f}%")
    print(f"{kv:7} " + " ".join(cells))

print("\n===== KV, bayt na to zhe okno =====")
if "bf16" in kvb:
    for kv in ORDER:
        if kv in kvb:
            print(f"{kv:7} {kvb[kv]:>12,} B  = {kvb[kv]/kvb['bf16']*100:5.1f}% ot bf16"
                  + (f", {(kvb[kv]/kvb['int8']-1)*100:+5.1f}% k int8" if "int8" in kvb else ""))
