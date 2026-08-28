#!/usr/bin/env python3
"""Stream both SASS dumps once, count instructions per function, and report only the functions
whose body actually changed between the two arms."""
import re, collections, sys

OPS = ("STS", "LDS", "LDG", "STG", "IMAD", "LOP3", "PRMT", "SHF", "HFMA2", "HMMA", "F2FP", "I2F")
FUNC = re.compile(r"\s*Function : (\S+)")
INSN = re.compile(r"\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9._]*)")


def norm(name):
    """nvcc gives internal-linkage kernels a per-compilation module id, so the same instantiation
    carries a different symbol in two builds. Match on the mangled body instead."""
    i = name.find("_ZN")
    return name[i:] if i > 0 else name

def scan(path):
    out = {}
    name, c, total = None, collections.Counter(), 0
    with open(path, errors="ignore") as f:
        for line in f:
            m = FUNC.match(line)
            if m:
                if name:
                    out[norm(name)] = (total, c)
                name, c, total = m.group(1), collections.Counter(), 0
            elif name is not None:
                m = INSN.search(line)
                if m:
                    op = m.group(1).split(".")[0]
                    total += 1
                    if op in OPS:
                        c[op] += 1
    if name:
        out[norm(name)] = (total, c)
    return out

a = scan(sys.argv[1])
b = scan(sys.argv[2])
print("functions: %d / %d" % (len(a), len(b)))
changed = [k for k in a if k in b and a[k][0] != b[k][0]]
print("changed bodies: %d" % len(changed))
for k in sorted(changed, key=lambda k: -abs(b[k][0] - a[k][0])):
    ta, tb = a[k][0], b[k][0]
    short = k[:100] + ("..." if len(k) > 100 else "")
    print("\n### %s" % short)
    print("  instructions %d -> %d (%+.1f%%)" % (ta, tb, (tb / ta - 1) * 100))
    line = "  "
    for op in OPS:
        if a[k][1][op] or b[k][1][op]:
            line += "%s %d->%d  " % (op, a[k][1][op], b[k][1][op])
    print(line)
