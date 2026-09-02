#!/usr/bin/env python3
"""Svodka perplexii po kodekam KV: obshchaya i po domenam, s otnosheniem k int8 i bf16."""
import json, pathlib, sys

D = pathlib.Path("/root/exp/kv/ppl")
ORDER = ["bf16", "int8", "fp8", "k8v4", "nvfp4"]

def find_report(d):
    for p in list(d.rglob("report.json")):
        return p
    return None

def extract(doc):
    """Vernut (overall, {domain: ppl}) iz report.json, kakim by ni byla ego forma."""
    overall, doms = None, {}
    def walk(o, key=None):
        nonlocal overall
        if isinstance(o, dict):
            if "perplexity" in o and isinstance(o["perplexity"], (int, float)):
                name = o.get("domain") or o.get("id") or o.get("name") or key
                if name in (None, "overall", "all", "total"):
                    overall = o["perplexity"]
                else:
                    doms[str(name)] = o["perplexity"]
            for k, v in o.items():
                if k == "overall" and isinstance(v, (int, float)):
                    overall = v
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
    walk(doc)
    return overall, doms

for proto, label in (("L", "polnyy korpus, kontekst 65536, shag 32768 -- protokol avtora dlya kodekov"),
                     ("S", "polnyy korpus, kontekst 4096, shag 2048 -- protokol po umolchaniyu")):
    rows = {}
    for kv in ORDER:
        d = D / f"{proto}_{kv}"
        if not d.exists():
            continue
        r = find_report(d)
        if not r:
            continue
        try:
            ov, dom = extract(json.loads(r.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  {kv}: ne razobran report.json ({e})"); continue
        if ov is not None:
            rows[kv] = (ov, dom)
    if not rows:
        print(f"\n===== {label}: NET DANNYH =====")
        continue
    print(f"\n===== {label} =====")
    base_i = rows.get("int8", (None,))[0]
    base_b = rows.get("bf16", (None,))[0]
    doms = sorted({d for _, dd in rows.values() for d in dd})
    print(f"{'kodek':8} {'obshchaya':>12} {'k int8':>10} {'k bf16':>10}  " +
          " ".join(f"{d[:18]:>19}" for d in doms))
    for kv in ORDER:
        if kv not in rows:
            continue
        ov, dom = rows[kv]
        di = f"{(ov/base_i-1)*100:+9.4f}%" if base_i else f"{'-':>10}"
        db = f"{(ov/base_b-1)*100:+9.4f}%" if base_b else f"{'-':>10}"
        cells = []
        for d in doms:
            v = dom.get(d)
            if v is None:
                cells.append(f"{'-':>19}"); continue
            bi = rows["int8"][1].get(d) if "int8" in rows else None
            cells.append(f"{v:9.4f}{(v/bi-1)*100:+8.3f}%" if bi else f"{v:19.4f}")
        print(f"{kv:8} {ov:12.5f} {di} {db}  " + " ".join(cells))
    print("  (bolshe -- huzhe; procenty k int8 v skobkah po domenam)")
