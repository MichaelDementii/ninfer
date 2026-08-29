#!/bin/bash
# The numerical half. The shipped test qualifies the materialising route at T=49 and 128 and the
# fused TMA route at T=1024, against an FP64 oracle with a criterion the file says belongs to the
# activation-compute profile rather than to a private materialised/fused implementation. It never
# covers 1280 and up - exactly the widths this change re-routes. Same test, same oracle, same
# criterion, two arms: on master those widths take the materialising path, on the branch the fused
# one. NINFER_OP_REPORT_STATS makes the harness print the residuals instead of only a verdict.
set -u
cd /root/ninfer_d4 || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
export NINFER_OP_REPORT_STATS=1
OUT=/root/qual/gNV
for arm in mst nv; do
  case $arm in
    mst) git checkout -q origin/master ;;
    nv)  git checkout -q perf/nvfp4-fused-swiglu-every-width ;;
  esac
  sed -i 's/std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024}/std::array<std::int32_t, 8> kA4Cases{5, 48, 49, 128, 1024, 1280, 2048, 4096}/' \
    tests/ops/linear_swiglu/test_nvfp4.cpp
  cmake --build build -j16 --target ninfer_linear_swiglu_nvfp4_test > /dev/null 2>&1
  echo "test build $arm rc=$?"
  timeout 3600 ./build/tests/ninfer_linear_swiglu_nvfp4_test > $OUT/oracle_stats_$arm.txt 2>&1
  echo "oracle $arm rc=$? lines=$(wc -l < $OUT/oracle_stats_$arm.txt)"
  git checkout -q -- tests/ops/linear_swiglu/test_nvfp4.cpp
done
git checkout -q perf/nvfp4-fused-swiglu-every-width

echo "=== A4 residuals against the FP64 oracle, by width and arm ==="
python3 - <<'PY'
import re, os
OUT = "/root/qual/gNV"
def scan(p):
    out = {}
    for line in open(p, errors="ignore"):
        if "OP_ERROR_STATS" not in line or "NVFP4_A4" not in line:
            continue
        rel = float(re.search(r"rel_l2=([0-9.eE+-]+)", line).group(1))
        lim = float(re.search(r"rel_l2_limit=([0-9.eE+-]+)", line).group(1))
        gr  = float(re.search(r"gross_ratio=([0-9.eE+-]+)", line).group(1))
        mx  = float(re.search(r"max_abs=([0-9.eE+-]+)", line).group(1))
        case = re.search(r"case=(.*)$", line).group(1).strip()
        m = re.search(r"[Tt]\s*=?\s*(\d+)", case)
        t = int(m.group(1)) if m else None
        out.setdefault(t, []).append((rel, lim, gr, mx, case))
    return out
A = scan(f"{OUT}/oracle_stats_mst.txt")
B = scan(f"{OUT}/oracle_stats_nv.txt")
print("%6s | %-34s | %-34s" % ("T", "master (route per master table)", "branch (fused where re-routed)"))
print("%6s | %12s %10s %8s | %12s %10s %8s" % ("", "rel_l2", "of limit", "gross", "rel_l2", "of limit", "gross"))
for t in sorted(set(A) | set(B), key=lambda x: (x is None, x)):
    for i in range(max(len(A.get(t, [])), len(B.get(t, [])))):
        a = A.get(t, [(None,)*5])[i] if i < len(A.get(t, [])) else (None,)*5
        b = B.get(t, [(None,)*5])[i] if i < len(B.get(t, [])) else (None,)*5
        fa = "%12.4g %9.1f%% %8.3f" % (a[0], 100*a[0]/a[1], a[2]) if a[0] is not None else " " * 32
        fb = "%12.4g %9.1f%% %8.3f" % (b[0], 100*b[0]/b[1], b[2]) if b[0] is not None else " " * 32
        better = ""
        if a[0] is not None and b[0] is not None:
            better = "  branch closer" if b[0] < a[0] else ("  same" if b[0] == a[0] else "  master closer")
        print("%6s | %s | %s%s" % (t, fa, fb, better))
PY
echo NV_ORACLE_DONE
