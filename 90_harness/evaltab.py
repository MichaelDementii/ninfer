#!/usr/bin/env python3
"""Svodka po prognam ocenki: kazhdaya ruka protiv drugoy, odin dataset -- odna stroka."""
import json, pathlib, sys, collections

RUNS = pathlib.Path("/root/ninfer_d4/eval/runs")
MAP  = pathlib.Path("/root/exp/kv/runmap.txt")
if not MAP.exists():
    sys.exit("net runmap.txt")

# tag -> run_id; tag vida "<suite>_<kv>"
res = collections.defaultdict(dict)   # dataset -> kv -> (metric, score, done, failed)
order = []
for line in MAP.read_text(encoding="utf-8").split("\n"):
    line = line.strip()
    if not line:
        continue
    tag, rid = line.split()
    kv = tag.rsplit("_", 1)[1]
    s = RUNS / rid / "summary.json"
    if not s.exists():
        print(f"  {tag}: net summary.json"); continue
    doc = json.loads(s.read_text(encoding="utf-8"))
    for job in doc.get("jobs", []):
        ds = job.get("dataset") or job.get("id")
        m  = job.get("primary_metric") or job.get("metric") or "?"
        sc = job.get("score")
        cp = job.get("completed"); fl = job.get("failed")
        if ds not in res:
            order.append(ds)
        res[ds][kv] = (m, sc, cp, fl)

ARMS = ["bf16", "int8", "fp8"]
print(f"{'dataset':22} {'metrika':22} " + " ".join(f"{a:>12}" for a in ARMS)
      + f" {'fp8-int8':>10} {'fp8-bf16':>10}")
for ds in order:
    row = res[ds]
    metric = next((v[0] for v in row.values()), "?")
    cells = []
    for a in ARMS:
        cells.append(f"{row[a][1]:12.4f}" if a in row and isinstance(row[a][1], (int, float)) else f"{'-':>12}")
    def d(x, y):
        if x in row and y in row and isinstance(row[x][1], (int, float)) and isinstance(row[y][1], (int, float)):
            return f"{(row[x][1]-row[y][1])*100:+10.2f}"
        return f"{'-':>10}"
    print(f"{ds:22} {metric:22} " + " ".join(cells) + f" {d('fp8','int8')} {d('fp8','bf16')}")

print("\n(raznica v punktah tochnosti; poloski proby)")
for ds in order:
    for a, v in sorted(res[ds].items()):
        print(f"  {ds:20} {a:6} vypolneno={v[2]} sboev={v[3]}")
