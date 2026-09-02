#!/usr/bin/env python3
"""Razbor zhurnalov ninfer-serve po chislu liniy.

Zapis request_done neset engine_timing: ozhidanie v ocheredi, ekspozirovannoe ozhidanie
ustroystva, hostovye nakladnye i chislo raundov dekoda. Imenno eti chetyre velichiny
i pokazyvayut, gde konkurentnost nachinaet stoit.
"""

import glob
import json
import os
import sys


def dig(o, *path):
    cur = o
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def summarize(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("event") == "request_done":
            rows.append(r)
    if not rows:
        return None
    out = {"n": len(rows)}
    fields = {
        "ochered": ("engine_timing", "queue_wait_seconds"),
        "ustroystvo": ("engine_timing", "device_wait_exposed_seconds"),
        "host": ("engine_timing", "host_exposed_seconds", "total"),
        "submit": ("engine_timing", "host_exposed_seconds", "program_submit"),
        "dek_ustr": ("engine_timing", "decode", "device_wait_exposed_seconds"),
        "dek_host": ("engine_timing", "decode", "host_exposed_seconds"),
        "raundov": ("engine_timing", "decode", "rounds"),
    }
    for name, path_keys in fields.items():
        vals = [dig(r, *path_keys) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            vals.sort()
            out[name] = (sum(vals) / len(vals), vals[len(vals) // 2], vals[-1])
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/root/exp/par25"
    files = sorted(glob.glob(os.path.join(base, "req_*.jsonl")))
    print("  %-14s %4s | %-24s %-24s %-24s %-20s" %
          ("ruka", "n", "ожид.очереди ср/мед/макс", "ожид.устройства", "хост всего", "раундов дек."))
    for f in files:
        tag = os.path.basename(f)[4:-6]
        s = summarize(f)
        if not s:
            print("  %-14s   -- net zapisey request_done" % tag)
            continue
        fmt = lambda k, p=4: (("%." + str(p) + "f/%." + str(p) + "f/%." + str(p) + "f") % s[k]) if k in s else "-"
        print("  %-14s %4d | %-24s %-24s %-24s %-20s" %
              (tag, s["n"], fmt("ochered"), fmt("ustroystvo"), fmt("host"), fmt("raundov", 1)))


if __name__ == "__main__":
    main()
