#!/usr/bin/env python3
"""Propusknaya sposobnost pod REALNOY nagruzkoy benchmarkov, iz zhurnalov zaprosov.

Tretiy zamer skorosti, nezavisimyy ot sinteticheskogo bencha: te zhe zaprosy, chetyre
linii, nastoyashchie dliny promptov i otvetov. Metriki:
  * ms/raund dekoda -- ne zavisit ot priyomki chernovika;
  * tok/s dekoda -- zavisit, poetomu priyomka pokazana ryadom;
  * prefill tok/s.
"""
import json, pathlib, statistics, sys, collections

D = pathlib.Path("/root/exp/kv/srv")

def summarize(path):
    ms_round, dec_tps, pre_tps, acc, out_toks = [], [], [], [], []
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        r = d.get("result") or {}
        t = d.get("timings_seconds") or {}
        s = d.get("speculative") or {}
        ct, dec = r.get("completion_tokens"), t.get("decode")
        if not ct or not dec or dec <= 0:
            continue
        rounds = s.get("rounds") or 0
        if rounds > 0:
            ms_round.append(1000.0 * dec / rounds)
        dec_tps.append(ct / dec)
        drafted, accepted = s.get("drafted_tokens") or 0, s.get("accepted_tokens") or 0
        if drafted:
            acc.append(accepted / drafted)
        pt, pre = r.get("computed_prefill_tokens"), t.get("prefill")
        if pt and pre and pre > 0:
            pre_tps.append(pt / pre)
        out_toks.append(ct)
    return ms_round, dec_tps, pre_tps, acc, out_toks

groups = collections.defaultdict(dict)
for p in sorted(D.glob("*.requests.jsonl")):
    tag = p.name.replace(".requests.jsonl", "")
    if "_" not in tag:
        continue
    bench, kv = tag.rsplit("_", 1)
    if kv not in ("bf16", "int8", "fp8", "nvfp4", "k8v4"):
        continue
    r = summarize(p)
    if r[0] or r[1]:
        groups[bench][kv] = r

for bench in sorted(groups):
    arms = groups[bench]
    if len(arms) < 2:
        continue
    print(f"\n===== {bench} =====")
    print(f"{'kodek':8} {'zaprosov':>9} {'ms/raund':>10} {'dekod tok/s':>12} {'prefill tok/s':>14} {'priyomka':>9} {'tok/otvet':>10}")
    base = None
    for kv in ("bf16", "int8", "fp8", "nvfp4", "k8v4"):
        if kv not in arms:
            continue
        ms, dt, pt, ac, ot = arms[kv]
        m = statistics.median(ms) if ms else float("nan")
        if kv == "int8":
            base = m
        d = statistics.median(dt) if dt else float("nan")
        p = statistics.median(pt) if pt else float("nan")
        a = statistics.mean(ac) if ac else float("nan")
        o = statistics.median(ot) if ot else 0
        rel = f" ({(base-m)/base*100:+.2f}% k int8)" if base and kv != "int8" else ""
        print(f"{kv:8} {len(ot):9d} {m:10.4f} {d:12.2f} {p:14.1f} {a:9.4f} {o:10.0f}{rel}")
