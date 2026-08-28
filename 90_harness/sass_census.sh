#!/bin/bash
# Full SASS census: which bodies changed, whether HMMA moves anywhere, and whether the bodies
# that keep their instruction count also keep their per-opcode census.
set -u
/usr/local/cuda/bin/cuobjdump -sass /root/bins/x_cli_mst > /root/s_a.txt 2>/dev/null
/usr/local/cuda/bin/cuobjdump -sass /root/bins/x_cli_w8  > /root/s_b.txt 2>/dev/null
python3 - <<'PY'
import re, collections
FUNC = re.compile(r"\s*Function : (\S+)")
INSN = re.compile(r"\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9._]*)")

def norm(n):
    i = n.find("_ZN")
    return n[i:] if i > 0 else n

def scan(path):
    out, name, c, total = {}, None, collections.Counter(), 0
    with open(path, errors="ignore") as f:
        for line in f:
            m = FUNC.match(line)
            if m:
                if name: out[norm(name)] = (total, c)
                name, c, total = m.group(1), collections.Counter(), 0
            elif name is not None:
                m = INSN.search(line)
                if m:
                    total += 1
                    c[m.group(1).split(".")[0]] += 1
    if name: out[norm(name)] = (total, c)
    return out

a, b = scan("/root/s_a.txt"), scan("/root/s_b.txt")
common = [k for k in a if k in b]
print("functions: %d / %d, comparable %d, only-in-master %d, only-in-branch %d"
      % (len(a), len(b), len(common), len(a) - len(common), len(b) - len(common)))
diff_total = [k for k in common if a[k][0] != b[k][0]]
diff_census = [k for k in common if a[k][1] != b[k][1]]
print("bodies with a different instruction count : %d" % len(diff_total))
print("bodies with a different per-opcode census : %d" % len(diff_census))
print("all of them are w8_rowsplit_gemm_mma_kernel: %s"
      % all("w8_rowsplit_gemm_mma_kernel" in k for k in diff_census))
hm = [(k, a[k][1]["HMMA"], b[k][1]["HMMA"]) for k in diff_census if a[k][1]["HMMA"] != b[k][1]["HMMA"]]
print("changed bodies whose HMMA count moves  : %d" % len(hm))
for op in ("STS", "LDS", "I2F", "F2FP", "SHFL", "HMMA", "IMAD", "LOP3"):
    sa = sum(a[k][1][op] for k in diff_census); sb = sum(b[k][1][op] for k in diff_census)
    print("  %-5s across the %d changed bodies: %6d -> %6d" % (op, len(diff_census), sa, sb))
d = sorted((b[k][0] / a[k][0] - 1) * 100 for k in diff_census)
print("instruction delta: %.1f%% .. %.1f%%, median %.1f%%" % (d[0], d[-1], d[len(d) // 2]))
PY
rm -f /root/s_a.txt /root/s_b.txt
echo CENSUS_DONE
