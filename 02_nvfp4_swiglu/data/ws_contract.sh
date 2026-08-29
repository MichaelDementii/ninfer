#!/bin/bash
# Does the change move the workspace capacity contract anywhere, not just at the two chunk widths
# the product run happened to use? The commit edits
# nvfp4_linear_swiglu_workspace_capacity_bytes, and workspace boundaries are contract, so this
# enumerates the function over a grid of (min_tokens, max_tokens) on both arms and diffs.
set -u
cd /root/ninfer_d4 || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
mkdir -p /root/scratch_ws
cat > /root/scratch_ws/probe.cpp <<'CPP'
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include <cstdio>
#include <vector>
using namespace ninfer;
using namespace ninfer::ops;
int main() {
    std::vector<int> pts;
    for (int t = 1; t <= 64; ++t) pts.push_back(t);
    for (int t : {96, 128, 200, 255, 256, 257, 512, 768, 1023, 1024, 1025, 1279, 1280, 1281,
                  1536, 2047, 2048, 2049, 3000, 4095, 4096, 4097, 6144, 8191, 8192, 8193,
                  12288, 16384, 32768, 65536, 131072})
        pts.push_back(t);
    for (int lo : pts) {
        for (int hi : pts) {
            if (hi < lo) continue;
            std::size_t v = 0;
            bool threw = false;
            try {
                v = detail::nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy::AllowA4, lo, hi);
            } catch (...) { threw = true; }
            std::printf("%d %d %s %zu\n", lo, hi, threw ? "THROW" : "OK", v);
        }
    }
    return 0;
}
CPP
FLAGS=$(python3 - <<'PY'
import json
db = json.load(open("/root/ninfer_d4/build/compile_commands.json"))
for e in db:
    if e["file"].endswith("nvfp4_linear_swiglu_plan.cpp"):
        parts = e["command"].split()
        keep = [p for p in parts if p.startswith("-I") or p.startswith("-D") or p.startswith("-std=")]
        print(" ".join(keep)); break
PY
)
for arm in mst nv; do
  case $arm in
    mst) git checkout -q origin/master ;;
    nv)  git checkout -q perf/nvfp4-fused-swiglu-every-width ;;
  esac
  cmake --build build -j16 --target ninfer_ops > /dev/null 2>&1
  g++ -O1 -std=c++20 -I/usr/local/cuda-13.1/include $FLAGS /root/scratch_ws/probe.cpp \
      -o /root/scratch_ws/probe_$arm \
      -Wl,--start-group build/src/libninfer_ops.a build/src/libninfer_core.a build/src/libninfer_nvfp4_tma.a build/src/libninfer_artifact.a build/src/libninfer_text.a -Wl,--end-group -L/usr/local/cuda-13.1/lib64 -lcudart -lcuda -lpthread 2>/root/scratch_ws/cc_$arm.log
  rc=$?; echo "compile $arm rc=$rc"; [ $rc = 0 ] || { tail -5 /root/scratch_ws/cc_$arm.log; continue; }
  /root/scratch_ws/probe_$arm > /root/scratch_ws/cap_$arm.txt 2>&1
  echo "run $arm rc=$? rows=$(wc -l < /root/scratch_ws/cap_$arm.txt)"
done
git checkout -q perf/nvfp4-fused-swiglu-every-width
echo "=== capacity contract diff ==="
if [ -f /root/scratch_ws/cap_mst.txt ] && [ -f /root/scratch_ws/cap_nv.txt ]; then
  python3 - <<'PY'
a = {}
for line in open("/root/scratch_ws/cap_mst.txt"):
    lo, hi, st, v = line.split(); a[(int(lo), int(hi))] = (st, int(v))
b = {}
for line in open("/root/scratch_ws/cap_nv.txt"):
    lo, hi, st, v = line.split(); b[(int(lo), int(hi))] = (st, int(v))
keys = sorted(set(a) & set(b))
diff = [k for k in keys if a[k] != b[k]]
print("intervals probed: %d" % len(keys))
print("intervals where the capacity or the throw behaviour differs: %d" % len(diff))
bigger = [k for k in diff if b[k][0] == "OK" and a[k][0] == "OK" and b[k][1] > a[k][1]]
smaller = [k for k in diff if b[k][0] == "OK" and a[k][0] == "OK" and b[k][1] < a[k][1]]
thrown = [k for k in diff if a[k][0] != b[k][0]]
print("  branch asks for MORE: %d    branch asks for LESS: %d    throw behaviour changed: %d"
      % (len(bigger), len(smaller), len(thrown)))
for label, ks in (("MORE", bigger), ("LESS", smaller), ("THROW", thrown)):
    for k in ks[:8]:
        print("   %-5s min=%-7d max=%-7d  master=%s %d  branch=%s %d"
              % (label, k[0], k[1], a[k][0], a[k][1], b[k][0], b[k][1]))
PY
fi
echo WS_CONTRACT_DONE
