#!/bin/bash
# What the NVFP4 submission still lacks: a suite run on both arms, a codegen census (the diff is a
# host-side route table, so nothing should move), and a boundary control. The fused route needs
# T >= 1024 AND T % 256 == 0, so widths in between must stay on the materialising path and read
# exactly 1.000 - the same built-in control the sweep above 1024 already gives.
set -u
cd /root/ninfer_d4 || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
export NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=/root/models/qwen3_8_27b_nvfp4.ninfer
export NINFER_QWEN3_8_27B_NVFP4_ARTIFACT=/root/models/qwen3_8_27b_nvfp4.ninfer
export NINFER_QWEN3_6_35B_A3B_WEIGHTS=/root/models/qwen3_6_35b_a3b.ninfer
export NINFER_QWEN3_6_35B_A3B_ARTIFACT=/root/models/qwen3_6_35b_a3b.ninfer
export NINFER_QWEN3_6_27B_WEIGHTS=/root/models/qwen3_6_27b.ninfer
export NINFER_QWEN3_6_27B_ARTIFACT=/root/models/qwen3_6_27b.ninfer
OUT=/root/qual/gNV

echo "########## boundary control: multiples of 256 versus not ##########"
for arm in mst nv; do
  /root/bins/n_swig_$arm --policy a4 \
    --t-sweep 1024,1025,1152,1279,1280,1281,1408,1536,1792,2048,2304,2560 --repeat 50 \
    --csv-out $OUT/bound_$arm.csv > /dev/null 2>&1
  echo "bound $arm rc=$?"
done

echo "########## ctest, both arms ##########"
for ref in origin/master perf/nvfp4-fused-swiglu-every-width; do
  git checkout -q "$ref"
  cmake --build build -j16 > /dev/null 2>&1
  echo "--- $ref ---"
  ( cd build && ctest -j1 2>&1 | tail -8 )
done
git checkout -q perf/nvfp4-fused-swiglu-every-width

echo "########## codegen census: a host route table should move nothing ##########"
/usr/local/cuda/bin/cuobjdump -sass /root/bins/n_cli_mst > /root/nv_a.txt 2>/dev/null
/usr/local/cuda/bin/cuobjdump -sass /root/bins/n_cli_nv  > /root/nv_b.txt 2>/dev/null
python3 /root/sass_diff.py /root/nv_a.txt /root/nv_b.txt 2>&1 | head -4
python3 /root/resu2.py /root/bins/n_cli_mst /root/bins/n_cli_nv 2>&1 | head -12
rm -f /root/nv_a.txt /root/nv_b.txt
echo NV_FINAL_DONE
