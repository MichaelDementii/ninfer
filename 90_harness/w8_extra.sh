#!/bin/bash
# Follow-up to w8_arms.sh, on the same two stashed builds.
#
# 1. The GROUPS == 4 schedule. w8_dispatch.cpp selects launch_w8_mma_r64x16_c48_k128_a1 at
#    k=5120, n=34816 for t in [41,48] and k=5120, n=248320 for t in [34,48]. Nothing in the
#    campaign samples those extents, and that is the branch carrying two of the three shfl_sync
#    sites. Sweep across both route boundaries instead of reading the dispatch table.
# 2. Repeatability. The campaign's single pass put three rows below unity at very short extents
#    (w8_linear_add decode T=1 at 0.857, linear_swiglu exact_t at T=3 and T=11). Those are on
#    kernels this branch does not change, so they are the harness floor rather than a result -
#    but that has to be measured, not asserted. Three independent passes per arm.
# 3. Updated prefill kernel breakdown, so the next candidate is picked from a profile taken
#    after the MoE half landed rather than before it.
set -u
export PATH=/usr/local/cuda-13.1/bin:$PATH
export NINFER_QWEN3_6_35B_A3B_WEIGHTS=/root/models/qwen3_6_35b_a3b.ninfer
export NINFER_QWEN3_6_35B_A3B_ARTIFACT=/root/models/qwen3_6_35b_a3b.ninfer
BINS=/root/bins
OUT=/root/qual/gW
mkdir -p $OUT/prof

echo "########## GROUPS == 4 extents ##########"
for arm in mst w8; do
  $BINS/x_lin_$arm --qtype W8 --n 34816  --k 5120 --sweep 30:56 --repeat 50 \
     --csv-out $OUT/g4_34816_${arm}.csv  > /dev/null 2>&1; echo "g4 34816 $arm rc=$?"
  $BINS/x_lin_$arm --qtype W8 --n 248320 --k 5120 --sweep 30:56 --repeat 50 \
     --csv-out $OUT/g4_248320_${arm}.csv > /dev/null 2>&1; echo "g4 248320 $arm rc=$?"
done

echo "########## repeatability: three passes per arm ##########"
for p in 1 2 3; do
  for arm in mst w8; do
    $BINS/x_add_$arm  --production-only --repeat 50 --csv-out $OUT/rep_add_${arm}_$p.csv  > /dev/null 2>&1
    $BINS/x_swig_$arm --production-only --repeat 50 --csv-out $OUT/rep_swig_${arm}_$p.csv > /dev/null 2>&1
    echo "pass $p $arm rc=$?"
  done
done

echo "########## prefill kernel breakdown, both arms ##########"
cd $OUT/prof
for tag in mst w8; do
  nsys profile -t cuda --cuda-graph-trace=node -o $OUT/prof/p_$tag --force-overwrite=true \
    $BINS/x_cli_$tag /root/models/qwen3_6_35b_a3b.ninfer \
    --messages /root/qual/fixtures/niah_16k.json --max-new 1 --greedy --no-thinking \
    --max-context 131072 --prefill-chunk 8192 --kv-dtype bf16 \
    > $OUT/prof/p_$tag.txt 2> $OUT/prof/p_$tag.err
  echo "nsys $tag rc=$?"
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    -o $OUT/prof/k_$tag $OUT/prof/p_$tag.nsys-rep > /dev/null 2>&1
  echo "stats $tag rc=$?"
done
rm -f $OUT/prof/*.sqlite
ls -la $OUT/prof
echo W8_EXTRA_DONE
