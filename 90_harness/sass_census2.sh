#!/bin/bash
# Correct SASS opcode census.
#
# The census in the campaign used /root/sass_diff.py, whose instruction regex
#     \*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9._]*)
# cannot parse a uniform predicate: on "@!UP0  LDS R4, ..." the `P?` group fails on the U, so the
# alternation falls through and the *predicate* UP0 is recorded as the opcode while the real one is
# dropped. Every opcode that ever appears under a uniform predicate is therefore undercounted, and
# LDS is the one that matters here. This parser strips predicate tokens explicitly instead.
set -u
/usr/local/cuda/bin/cuobjdump -sass /root/bins/y_cli_mst > /root/c_a.txt 2>/dev/null
/usr/local/cuda/bin/cuobjdump -sass /root/bins/y_cli_w8  > /root/c_b.txt 2>/dev/null
python3 - <<'PY'
import re, collections
FUNC = re.compile(r"\s*Function : (\S+)")
PRED = re.compile(r"^@!?U?P\d+\s+")

def norm(n):
    i = n.find("_ZN")
    return n[i:] if i > 0 else n

def opcode(line):
    # everything after the address comment, with the encoding comment and the trailing ; removed
    i = line.find("*/")
    if i < 0:
        return None
    t = line[i + 2:]
    j = t.find("/*")
    if j >= 0:
        t = t[:j]
    t = t.strip()
    while True:
        m = PRED.match(t)
        if not m:
            break
        t = t[m.end():]
    m = re.match(r"([A-Z][A-Z0-9]*)", t)
    return m.group(1) if m else None

def scan(path):
    out, name, c, total = {}, None, collections.Counter(), 0
    with open(path, errors="ignore") as f:
        for line in f:
            m = FUNC.match(line)
            if m:
                if name:
                    out[norm(name)] = (total, c)
                name, c, total = m.group(1), collections.Counter(), 0
            elif name is not None:
                op = opcode(line)
                if op:
                    total += 1
                    c[op] += 1
    if name:
        out[norm(name)] = (total, c)
    return out

a, b = scan("/root/c_a.txt"), scan("/root/c_b.txt")
common = [k for k in a if k in b]
changed = [k for k in common if a[k] != b[k]]
print("functions %d / %d, comparable %d, unmatched %d" % (len(a), len(b), len(common), len(a)-len(common)))
print("bodies differing in count or per-opcode census: %d" % len(changed))
print("all of them this kernel: %s" % all("w8_rowsplit_gemm_mma_kernel" in k for k in changed))
tot_a = sum(a[k][0] for k in changed); tot_b = sum(b[k][0] for k in changed)
print("instructions over the changed bodies: %d -> %d (%+.1f%%)" % (tot_a, tot_b, (tot_b/tot_a-1)*100))
d = sorted((b[k][0]/a[k][0]-1)*100 for k in changed)
print("per-body delta: %.1f%% .. %.1f%%, median %.1f%%" % (d[0], d[-1], d[len(d)//2]))
ops = sorted(set(list(a[changed[0]][1]) + list(b[changed[0]][1])) |
             {o for k in changed for o in list(a[k][1]) + list(b[k][1])})
print("\nopcode sums over the %d changed bodies (only those that move):" % len(changed))
for op in ops:
    sa = sum(a[k][1][op] for k in changed); sb = sum(b[k][1][op] for k in changed)
    if sa != sb:
        print("  %-14s %7d -> %7d   %+.1f%%" % (op, sa, sb, (sb/sa-1)*100 if sa else float("inf")))
print("\nopcodes identical across all changed bodies:")
same = [op for op in ops if sum(a[k][1][op] for k in changed) == sum(b[k][1][op] for k in changed)
        and sum(a[k][1][op] for k in changed) > 0]
print("  " + ", ".join("%s %d" % (op, sum(a[k][1][op] for k in changed)) for op in same))
moved = [k for k in changed if a[k][1]["HMMA"] != b[k][1]["HMMA"]]
print("\nbodies where HMMA moves: %d" % len(moved))
big = max(changed, key=lambda k: a[k][0])
print("\nlargest changed body: %d -> %d (%+.1f%%)" % (a[big][0], b[big][0], (b[big][0]/a[big][0]-1)*100))
print("  " + big[:150])
deep = min(changed, key=lambda k: b[k][0]/a[k][0])
print("deepest reduction: %d -> %d (%+.1f%%)" % (a[deep][0], b[deep][0], (b[deep][0]/a[deep][0]-1)*100))
print("  " + deep[:150])
for k, lab in ((big, "largest"), (deep, "deepest")):
    print("  %s: " % lab + "  ".join("%s %d->%d" % (o, a[k][1][o], b[k][1][o])
          for o in ("STS","LDS","I2F","I2FP","F2FP","HMMA","SHFL","WARPSYNC") if a[k][1][o] or b[k][1][o]))
PY
rm -f /root/c_a.txt /root/c_b.txt
echo CENSUS2_DONE
