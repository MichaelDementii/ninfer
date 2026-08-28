#!/usr/bin/env python3
"""REG / SHARED / LOCAL for the touched kernels, correct field names this time."""
import re, subprocess, sys
CUOBJ = "/usr/local/cuda/bin/cuobjdump"
FUNC = re.compile(r"Function (\S+):")
LINE = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


def norm(name):
    """nvcc gives internal-linkage kernels a per-compilation module id, so the same instantiation
    carries a different symbol in two builds. Match on the mangled body instead."""
    i = name.find("_ZN")
    return name[i:] if i > 0 else name

def scan(path):
    out, name = {}, None
    txt = subprocess.run([CUOBJ, "--dump-resource-usage", path], capture_output=True, text=True).stdout
    for line in txt.splitlines():
        m = FUNC.search(line)
        if m:
            name = m.group(1); continue
        if name:
            m = LINE.search(line)
            if m:
                out[norm(name)] = tuple(int(x) for x in m.groups())
            name = None
    return out

a, b = scan(sys.argv[1]), scan(sys.argv[2])
for label, pat in (("w8_rowsplit_gemm_mma", "w8_rowsplit_gemm_mma"), ("sparse_moe_prefill", "sparse_moe_prefill")):
    keys = [k for k in a if k in b and pat in k]
    if not keys:
        print(label, "none"); continue
    ra = [a[k][0] for k in keys]; rb = [b[k][0] for k in keys]
    sa = [a[k][2] for k in keys]; sb = [b[k][2] for k in keys]
    la = [a[k][3] for k in keys]; lb = [b[k][3] for k in keys]
    stka = [a[k][1] for k in keys]; stkb = [b[k][1] for k in keys]
    print("\n=== %s: %d instantiations ===" % (label, len(keys)))
    print("  registers  max %d -> %d   mean %.1f -> %.1f   raised in %d, lowered in %d" %
          (max(ra), max(rb), sum(ra)/len(ra), sum(rb)/len(rb),
           sum(1 for k in keys if b[k][0] > a[k][0]), sum(1 for k in keys if b[k][0] < a[k][0])))
    print("  shared     %s -> %s" % (sorted(set(sa)), sorted(set(sb))))
    print("  local/spill %s -> %s   stack %s -> %s" % (sorted(set(la)), sorted(set(lb)), sorted(set(stka)), sorted(set(stkb))))
