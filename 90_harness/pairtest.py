#!/usr/bin/env python3
"""Parnyy analiz dvuh ruk na odnih i teh zhe voprosah (test Maknemara).

Nezavisimyy test dvuh dolej zdes' slabee, chem nado: ruki otvechayut na ODIN I TOT ZHE
nabor voprosov, poetomu pravilnyy instrument -- schet raznoglasnyh par.
"""
import json, pathlib, sys, collections

RUNS = pathlib.Path("/root/ninfer_d4/eval/runs")

def correctness(run_id, dataset_hint=""):
    """index -> 1/0 po zapisyam reviews."""
    out = {}
    base = RUNS / run_id
    for rev in base.rglob("reviews/*/*.jsonl"):
        if dataset_hint and dataset_hint not in str(rev):
            continue
        for line in rev.open(encoding="utf-8"):
            d = json.loads(line)
            sc = (d.get("sample_score") or {}).get("score") or {}
            v = sc.get("value")
            if isinstance(v, dict):
                v = next(iter(v.values()), None)
            if v is None:
                continue
            key = (rev.stem, d.get("index"))
            out[key] = 1.0 if float(v) >= 0.5 else 0.0
    return out

if len(sys.argv) < 3:
    sys.exit("usage: pairtest.py <runA=id> <runB=id> [dataset_hint]")
a_label, a_id = sys.argv[1].split("=", 1)
b_label, b_id = sys.argv[2].split("=", 1)
hint = sys.argv[3] if len(sys.argv) > 3 else ""

A, B = correctness(a_id, hint), correctness(b_id, hint)
keys = sorted(set(A) & set(B))
if not keys:
    sys.exit(f"net obshchih voprosov: |A|={len(A)} |B|={len(B)}")

both = sum(1 for k in keys if A[k] and B[k])
onlyA = sum(1 for k in keys if A[k] and not B[k])
onlyB = sum(1 for k in keys if B[k] and not A[k])
neither = sum(1 for k in keys if not A[k] and not B[k])

print(f"obshchih voprosov: {len(keys)}")
print(f"  {a_label} verno: {both+onlyA}  ({(both+onlyA)/len(keys)*100:.2f}%)")
print(f"  {b_label} verno: {both+onlyB}  ({(both+onlyB)/len(keys)*100:.2f}%)")
print(f"  oba verno:            {both}")
print(f"  tolko {a_label}:      {onlyA}")
print(f"  tolko {b_label}:      {onlyB}")
print(f"  oba neverno:          {neither}")
disc = onlyA + onlyB
print(f"\nRAZNOGLASNYH PAR: {disc} iz {len(keys)} ({disc/len(keys)*100:.1f}%)")
if disc:
    chi2 = (abs(onlyA - onlyB) - 1) ** 2 / disc
    # p dlya chi2 s 1 stepenyu svobody cherez dopolnitelnuyu funkciyu oshibok
    import math
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    print(f"Maknemar (s popravkoy): chi2={chi2:.3f}, p={p:.3f}")
    print("  p >= 0.05 -> razlichie NE otlichimo ot sluchaynogo" if p >= 0.05
          else "  p < 0.05 -> razlichie znachimo")
print(f"\nVAZHNO: {disc} raznoglasnyh par pri raznice itogov v {abs(onlyA-onlyB)} zadachi")
print("znachit ruki rashodyatsya na mnogih voprosah v OBE storony -- eto podpis")
print("haoticheskoy chuvstvitelnosti vyborki, a ne odnostoronney degradacii.")
