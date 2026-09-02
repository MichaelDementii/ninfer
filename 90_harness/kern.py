#!/usr/bin/env python3
"""Razbor nsys cuda_gpu_kern_sum. Imya ochishchaetsya do korotkogo, no ne do pustogo:
'(anonymous namespace)::' snimaetsya PERED tem, kak rezat po skobke."""

import csv
import io
import os
import sys

JUNK = ("(anonymous namespace)::", "ninfer::ops::detail::", "ninfer::ops::", "ninfer::",
        "detail::", "void ")


def short(name):
    n = name
    for j in JUNK:
        n = n.replace(j, "")
    n = n.split("<")[0].split("(")[0].strip()
    n = n.replace("sparse_moe_prefill_", "moe_pre_").replace("sparse_moe_", "moe_")
    n = n.replace("gated_delta_net::", "gdn::").replace("chunked::", "")
    for seg in ("state_passing::", "prepare_wy_wu::", "output::", "delta_rule::", "wy_fast::"):
        n = n.replace(seg, "")
    return n[-44:]


def load(path):
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return {}
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    try:
        s = next(i for i, l in enumerate(lines) if l.lstrip('"').startswith("Time"))
    except StopIteration:
        return {}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[s:]))))
    if not rows:
        return {}
    kt = next(k for k in rows[0] if "Total Time" in k)
    ki = next(k for k in rows[0] if "Instances" in k)
    out = {}
    for r in rows:
        k = short(r["Name"])
        t, c = out.get(k, (0.0, 0))
        out[k] = (t + float(r[kt]) / 1e6, c + int(float(r[ki])))
    return out


def table(d, title, top=26):
    if not d:
        print("  %s: net dannyh" % title)
        return
    tot = sum(v[0] for v in d.values())
    print("\n  ##### %s ##### GPU vsego = %.1f ms, puskov = %d, raznyh yader = %d" %
          (title, tot, sum(v[1] for v in d.values()), len(d)))
    print("  %-44s %9s %8s %9s %10s" % ("yadro", "ms", "%", "puskov", "mks/pusk"))
    for k, (t, c) in sorted(d.items(), key=lambda kv: -kv[1][0])[:top]:
        print("  %-44s %9.2f %7.2f%% %9d %10.1f" % (k, t, 100 * t / tot, c, 1000 * t / max(c, 1)))


def compare(a, b, na, nb, title, floor=2.0):
    keys = set(a) | set(b)
    print("\n  ##### %s #####" % title)
    ta, tb = sum(v[0] for v in a.values()), sum(v[0] for v in b.values())
    print("  itogo: %s = %.1f ms, %s = %.1f ms, delta = %+.2f%%" %
          (na, ta, nb, tb, 100 * (tb / ta - 1) if ta else 0))
    print("  %-44s %10s %10s %9s | %8s %8s" % ("yadro", na, nb, "delta", "n_" + na, "n_" + nb))
    rows = []
    for k in keys:
        x, y = a.get(k, (0.0, 0)), b.get(k, (0.0, 0))
        if max(x[0], y[0]) < floor:
            continue
        rows.append((abs(y[0] - x[0]), k, x, y))
    for _, k, x, y in sorted(rows, reverse=True):
        pc = 100 * (y[0] / x[0] - 1) if x[0] > 0 else float("inf")
        print("  %-44s %10.2f %10.2f %+8.2f%% | %8d %8d" % (k, x[0], y[0], pc, x[1], y[1]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "one":
        table(load(sys.argv[2]), sys.argv[3])
    elif mode == "cmp":
        compare(load(sys.argv[2]), load(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
