#!/bin/bash
# NVFP4 fused-SwiGLU route registration: the whole evidence campaign.
#
# The change is one line of a route table: the fused TMA SwiGLU was registered at exactly
# T == 1024 although the kernel accepts any T % 256 == 0, so every wider prefill chunk fell into
# the materialising path - a linear writing 34816xT bf16 into the arena and a separate silu_mul
# reading it back. This measures the operator, the arena, the numerics against the project's own
# FP64 oracle at the widths the change newly routes, and the product.
set -u
cd /root/ninfer_d4 || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
export NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=/root/models/qwen3_8_27b_nvfp4.ninfer
export NINFER_QWEN3_8_27B_NVFP4_ARTIFACT=/root/models/qwen3_8_27b_nvfp4.ninfer
BINS=/root/bins
OUT=/root/qual/gNV
rm -rf $OUT && mkdir -p $OUT
M=/root/models/qwen3_8_27b_nvfp4.ninfer

echo "########## 1. operator, both arms ##########"
for arm in mst nv; do
  $BINS/n_swig_$arm --policy a4 --t-sweep 256,512,768,1024,1280,2048,4096,8192 --repeat 50 \
    --csv-out $OUT/swig_$arm.csv > $OUT/swig_$arm.txt 2>&1
  echo "swiglu $arm rc=$?"
done

echo "########## 2. numerics against the shipped FP64 oracle ##########"
# The shipped test qualifies T=1024 on the fused route and T=49/128 on the materialising one.
# It never covers 2048 and up - exactly the widths this change re-routes. Same test, same oracle,
# same A4 criterion; on master those widths take the materialising path and on the branch the
# fused one, so the two arms answer the question directly.
cp tests/ops/linear_swiglu/test_nvfp4.cpp /tmp/test_nvfp4.cpp.orig
for arm in mst nv; do
  case $arm in
    mst) git checkout -q origin/master ;;
    nv)  git checkout -q perf/nvfp4-fused-swiglu-every-width ;;
  esac
  sed -i 's/std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024}/std::array<std::int32_t, 8> kA4Cases{5, 48, 49, 128, 1024, 1280, 2048, 4096}/' \
    tests/ops/linear_swiglu/test_nvfp4.cpp
  grep -n "kA4Cases{" tests/ops/linear_swiglu/test_nvfp4.cpp
  cmake --build build -j16 --target ninfer_linear_swiglu_nvfp4_test > /root/nv_test_build_$arm.log 2>&1
  echo "test build $arm rc=$?"; tail -2 /root/nv_test_build_$arm.log
  timeout 3600 ./build/tests/ninfer_linear_swiglu_nvfp4_test > $OUT/oracle_$arm.txt 2>&1
  echo "oracle $arm rc=$?"
  git checkout -q -- tests/ops/linear_swiglu/test_nvfp4.cpp
done

echo "########## 3. arena and workspace, from the product run summary ##########"
for arm in mst nv; do
  for ck in 1024 8192; do
    $BINS/n_cli_$arm "$M" --messages /root/qual/fixtures/niah_16k.json --max-new 8 --greedy \
      --no-thinking --max-context 131072 --prefill-chunk $ck --kv-dtype bf16 \
      > $OUT/ws_${arm}_$ck.txt 2> $OUT/ws_${arm}_$ck.err
    echo "ws $arm ck=$ck rc=$?"
  done
done

echo "########## 4. end to end, arms alternating ##########"
run() {
  local l=$1 r=$2 b=$3 p=$4 ck=$5 tag=$6
  local base=$OUT/${tag}_${l}_r${r}
  $b "$M" --messages "$p" --max-new 32 --greedy --no-thinking --max-context 131072 \
     --prefill-chunk "$ck" --kv-dtype bf16 > "$base.txt" 2> "$base.err"
  echo "$tag $l r$r prefill=$(grep -oE 'prefill speed +[0-9.]+' "$base.err" | grep -oE '[0-9.]+$') decode=$(grep -oE 'decode speed +[0-9.]+' "$base.err" | grep -oE '[0-9.]+$')"
}
pair_run() {
  local r=$1 p=$2 ck=$3 tag=$4
  if [ $((r % 2)) -eq 1 ]; then
    run MST "$r" $BINS/n_cli_mst "$p" "$ck" "$tag"; run NV "$r" $BINS/n_cli_nv "$p" "$ck" "$tag"
  else
    run NV "$r" $BINS/n_cli_nv "$p" "$ck" "$tag"; run MST "$r" $BINS/n_cli_mst "$p" "$ck" "$tag"
  fi
}
for r in 1 2 3 4; do
  pair_run "$r" /root/qual/fixtures/niah_16k.json 8192 "c8192_16k"
  pair_run "$r" /root/qual/fixtures/niah_32k.json 8192 "c8192_32k"
  pair_run "$r" /root/qual/fixtures/niah_16k.json 1024 "c1024_16k"
done

echo "########## 5. output comparison ##########"
# At chunk 1024 nothing about the route changes, so those must be byte-identical. At 8192 the
# route does change and the two epilogues round differently by construction, so they need not be.
for f in $(cd $OUT && ls *_MST_*.txt 2>/dev/null); do
  g=${f/_MST_/_NV_}
  [ -f "$OUT/$g" ] || continue
  if cmp -s "$OUT/$f" "$OUT/$g"; then echo "IDENTICAL $g"; else echo "DIFFERS   $g"; fi
done

echo "########## 6. which route each width takes, both arms ##########"
grep -c . $OUT/swig_mst.csv $OUT/swig_nv.csv 2>/dev/null
echo NV_CAMPAIGN_DONE
