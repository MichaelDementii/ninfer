#!/usr/bin/env python3
"""Doskoring NIAH po sohranennym predskazaniyam.

Shtatnyy sudya EvalScope trebuet doslovnogo sovpadeniya vsey frazy-igolki. Nasha model
otvechaet ee soderzhatelnoy chastyu i ne povtoryaet slova, uzhe stoyashchie v voprose,
poetomu shtatnaya metrika daet nol na OBEIH rukah -- razlichit ih nechem.

Sobstvennye metriki, odinakovye dlya vseh ruk:
  * polnota  -- dolya soderzhatelnyh tokenov igolki, naydennyh v otvete (nepreryvnaya);
  * popadanie -- polnota >= 0.7 (binarnaya);
  * F1       -- tokennoe F1 otveta protiv igolki.
Dlya CJK razbor posimvolnyy: probelov tam net.
Shtatnaya metrika tozhe pechataetsya ryadom -- nichego ne podmenyaetsya, dobavlyaetsya.
"""
import json, pathlib, re, sys, collections

STOP = {"the","a","an","is","to","in","on","and","of","do","that","for","with","be","as","at","by","it"}
CJK = re.compile(r"[㐀-鿿぀-ヿ]")

def toks(s):
    if CJK.search(s):
        return [c for c in re.sub(r"\s+", "", s) if not re.match(r"[^\w]", c)]
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return [t for t in s.split() if t]

def content(s):
    t = toks(s)
    return t if CJK.search(s) else [x for x in t if x not in STOP]

def f1(pred, gold):
    p, g = toks(pred), toks(gold)
    if not p or not g:
        return 0.0
    cp, cg = collections.Counter(p), collections.Counter(g)
    common = sum((cp & cg).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)

def recall(pred, gold, question=""):
    """Polnota po tokenam igolki, kotoryh NET v voprose.

    Slova, uzhe stoyashchie v voprose ("best thing", "San Francisco"), model
    zakonomerno ne povtoryaet, i trebovat ih -- znachit merit vezhlivost, a ne poisk.
    """
    cg = content(gold)
    if question:
        qs = set(toks(question))
        cg2 = [w for w in cg if w not in qs]
        if cg2:
            cg = cg2
    if not cg:
        return 0.0
    tp = collections.Counter(toks(pred))
    return sum(1 for w in cg if tp.get(w, 0) > 0) / len(cg)


QRE = __import__("re").compile(r"<question>\s*(.*?)\s*</question>", __import__("re").S)


def question_of(rec):
    for m in rec.get("messages") or []:
        if m.get("role") == "user":
            g = QRE.search(m.get("content") or "")
            if g:
                return g.group(1)
    return ""

def score_run(run_dir):
    rows = []
    for pf in pathlib.Path(run_dir).rglob("predictions/*/*.jsonl"):
        subset = pf.stem
        rev = pathlib.Path(str(pf).replace("/predictions/", "/reviews/"))
        golds, meta = {}, {}
        if rev.exists():
            for line in rev.open(encoding="utf-8"):
                d = json.loads(line)
                golds[d["index"]] = d.get("target", "")
                sc = d.get("sample_score") or {}
                m = sc.get("sample_metadata") or {}
                vals = list((sc.get("score") or {}).get("value", {}).values())
                meta[d["index"]] = (m.get("context_length"), m.get("depth_percent"), vals)
        for line in pf.open(encoding="utf-8"):
            d = json.loads(line)
            i = d["index"]
            try:
                pred = d["model_output"]["choices"][0]["message"]["content"] or ""
            except Exception:
                pred = ""
            gold = golds.get(i, "")
            if not gold:
                continue
            r = recall(pred, gold, question_of(d))
            ctx, depth, stock = meta.get(i, (None, None, [0.0]))
            rows.append({"subset": subset, "ctx": ctx, "depth": depth,
                         "recall": r, "hit": 1.0 if r >= 0.7 else 0.0,
                         "f1": f1(pred, gold), "stock": float(stock[0]) if stock else 0.0,
                         "len": len(pred)})
    return rows

if __name__ == "__main__":
    entries = []
    if len(sys.argv) > 1:
        entries = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:]]
    else:
        mp = pathlib.Path("/root/exp/kv/runmap.txt")
        if mp.exists():
            for line in mp.read_text(encoding="utf-8").split("\n"):
                if line.strip() and any(k in line for k in ("needle", "standard", "long_")):
                    tag, rid = line.split()
                    entries.append((tag, f"/root/ninfer_d4/eval/runs/{rid}"))
    if not entries:
        sys.exit("nechego schitat")

    per = {}
    print(f"{'ruka':24} {'prob':>5} {'popadanie':>10} {'polnota':>9} {'F1':>8} {'shtatnaya':>10}")
    for tag, rd in entries:
        rows = score_run(rd)
        if not rows:
            print(f"{tag:24} {'NET':>5}"); continue
        per[tag] = rows
        n = len(rows)
        g = lambda k: sum(r[k] for r in rows) / n
        print(f"{tag:24} {n:5d} {g('hit'):10.4f} {g('recall'):9.4f} {g('f1'):8.4f} {g('stock'):10.4f}")

    ctxs = sorted({r["ctx"] for rows in per.values() for r in rows if r["ctx"] is not None})
    if ctxs and len(per) > 1:
        print("\npopadanie po dline konteksta:")
        print(f"{'ruka':24} " + " ".join(f"{c:>8}" for c in ctxs))
        for tag, rows in per.items():
            cells = []
            for c in ctxs:
                sel = [r["hit"] for r in rows if r["ctx"] == c]
                cells.append(f"{sum(sel)/len(sel):8.3f}" if sel else f"{'-':>8}")
            print(f"{tag:24} " + " ".join(cells))
        print("\npolnota po glubine zalozheniya igolki:")
        deps = sorted({r["depth"] for rows in per.values() for r in rows if r["depth"] is not None})
        print(f"{'ruka':24} " + " ".join(f"{d:>6}" for d in deps))
        for tag, rows in per.items():
            cells = []
            for dd in deps:
                sel = [r["recall"] for r in rows if r["depth"] == dd]
                cells.append(f"{sum(sel)/len(sel):6.3f}" if sel else f"{'-':>6}")
            print(f"{tag:24} " + " ".join(cells))
