#!/bin/bash
# The submission says 1024 is the lower bound "because that is where the fused path starts winning".
# No measurement of the fused route below 1024 exists in the campaign - the 1.000 rows at 256/512/768
# time the materialising route on both arms. A reviewer built a probe and measured otherwise. Verify
# it independently, three runs, and then look at the numerics at those widths, because if the fused
# route does win below 1024 the honest thing is to say so and offer the lower bound rather than
# assert a reason that was never measured.
set -u
cd /root/ninfer_d4 || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
OUT=/root/qual/gNV
echo "########## 1. the fused route below 1024, three runs ##########"
for i in 1 2 3; do
  echo "--- run $i ---"
  /root/scratch_nvrev/probe_lb 2>&1 | tail -12
done | tee $OUT/lower_bound_probe.txt | grep -E "^(---|T=|[0-9])" | head -40

echo "########## 2. numerics at those widths, both arms ##########"
# A variant of the branch whose only difference is the lower bound, so the oracle can be asked
# whether the fused route meets the A4 criterion at 256/512/768 as well.
git checkout -q perf/nvfp4-fused-swiglu-every-width
git branch -f probe/nvfp4-lower-bound perf/nvfp4-fused-swiglu-every-width
git checkout -q probe/nvfp4-lower-bound
sed -i 's/if (tokens >= kPrimaryT \&\& (tokens % kTmaBlockM) == 0)/if (tokens >= kTmaBlockM \&\& (tokens % kTmaBlockM) == 0)/' \
  src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
grep -n "tokens >= kTmaBlockM" src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
sed -i 's/std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024}/std::array<std::int32_t, 8> kA4Cases{5, 48, 49, 128, 256, 512, 768, 1024}/' \
  tests/ops/linear_swiglu/test_nvfp4.cpp
cmake --build build -j16 --target ninfer_linear_swiglu_nvfp4_test > /dev/null 2>&1
echo "build rc=$?"
NINFER_OP_REPORT_STATS=1 timeout 1800 ./build/tests/ninfer_linear_swiglu_nvfp4_test \
  > $OUT/oracle_lower_nv.txt 2>&1
echo "oracle lowered rc=$? verdict=$(tail -1 $OUT/oracle_lower_nv.txt)"
git checkout -q -- tests/ops/linear_swiglu/test_nvfp4.cpp src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp

# and the same widths on master, where they take the materialising route
git checkout -q origin/master
sed -i 's/std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024}/std::array<std::int32_t, 8> kA4Cases{5, 48, 49, 128, 256, 512, 768, 1024}/' \
  tests/ops/linear_swiglu/test_nvfp4.cpp
cmake --build build -j16 --target ninfer_linear_swiglu_nvfp4_test > /dev/null 2>&1
NINFER_OP_REPORT_STATS=1 timeout 1800 ./build/tests/ninfer_linear_swiglu_nvfp4_test \
  > $OUT/oracle_lower_mst.txt 2>&1
echo "oracle master rc=$? verdict=$(tail -1 $OUT/oracle_lower_mst.txt)"
git checkout -q -- tests/ops/linear_swiglu/test_nvfp4.cpp
git checkout -q perf/nvfp4-fused-swiglu-every-width

echo "########## 3. side by side ##########"
python3 - <<'PY'
import re
def scan(p):
    out={}
    for line in open(p,errors="ignore"):
        if "OP_ERROR_STATS" not in line or "NVFP4_A4" not in line: continue
        rel=float(re.search(r"rel_l2=([0-9.eE+-]+)",line).group(1))
        gr=float(re.search(r"gross_ratio=([0-9.eE+-]+)",line).group(1))
        c=re.search(r"case=(.*)$",line).group(1).strip()
        m=re.search(r"[Tt]\s*=?\s*(\d+)",c)
        if m: out[int(m.group(1))]=(rel,gr)
    return out
A=scan("/root/qual/gNV/oracle_lower_mst.txt"); B=scan("/root/qual/gNV/oracle_lower_nv.txt")
print("%6s | %11s %8s | %11s %8s | %s"%("T","mst rel_l2","gross","low rel_l2","gross","verdict"))
for t in sorted(set(A)|set(B)):
    a=A.get(t); b=B.get(t)
    if a and b:
        v = "same" if abs(a[0]-b[0])<1e-12 else ("lowered closer" if b[0]<a[0] else "master closer")
        print("%6d | %11.6f %8.3f | %11.6f %8.3f | %s"%(t,a[0],a[1],b[0],b[1],v))
PY
echo NV_LOWER_DONE
