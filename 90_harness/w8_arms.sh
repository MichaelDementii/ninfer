#!/bin/bash
# Two arms from one build directory: unmodified master (3a61ef3f) and the W8 row-split decode
# widening alone. Master already carries the MoE half (merged as #106), so this campaign measures
# the projection half against the tree it will actually ship on.
set -u
cd /root/ninfer_d4 || exit 1
export PATH=/usr/local/cuda-13.1/bin:$PATH
export NINFER_QWEN3_6_35B_A3B_WEIGHTS=/root/models/qwen3_6_35b_a3b.ninfer
export NINFER_QWEN3_6_35B_A3B_ARTIFACT=/root/models/qwen3_6_35b_a3b.ninfer
export NINFER_QWEN3_6_27B_WEIGHTS=/root/models/qwen3_6_27b.ninfer
export NINFER_QWEN3_6_27B_ARTIFACT=/root/models/qwen3_6_27b.ninfer
export NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=/root/models/qwen3_8_27b_nvfp4.ninfer

BINS=/root/bins
OUT=/root/qual/gW
rm -rf $OUT && mkdir -p $OUT

stash () {
  s=$1
  for pair in "apps/ninfer:cli" "bench/ninfer_sparse_moe_bench:smoe" \
              "bench/ninfer_w8_linear_add_bench:add" "bench/ninfer_linear_bench:lin" \
              "bench/ninfer_linear_pair_bench:pair" "bench/ninfer_w8_linear_swiglu_bench:swig" \
              "bench/ninfer_gdn_input_proj_bench:gdnip" "bench/ninfer_attn_input_proj_bench:attnip"; do
    cp "build/${pair%%:*}" "$BINS/x_${pair##*:}_$s" || return 1
  done
  echo "STASHED $s at $(git rev-parse --short HEAD) $(git log --format=%s -1)"
}

build_arm () {   # ref tag
  git checkout -q "$1" || return 1
  cmake --build build -j16 > /root/x_build_$2.log 2>&1
  local rc=$?
  echo "BUILD $2 EXIT=$rc"; tail -2 /root/x_build_$2.log
  [ $rc = 0 ] || return 1
  stash "$2"
}

build_arm origin/master mst || exit 1
build_arm perf/widen-w8-rowsplit-decode w8 || exit 1

echo "########## ctest, both arms ##########"
for ref in origin/master perf/widen-w8-rowsplit-decode; do
  git checkout -q "$ref"
  cmake --build build -j16 > /dev/null 2>&1
  echo "--- $ref ---"
  ( cd build && ctest -j1 2>&1 | tail -25 ) 
done

echo "########## operator benchmarks, both arms ##########"
for arm in mst w8; do
  for tk in 1024 4096 8192; do
    $BINS/x_smoe_$arm --codec all --tokens $tk --cache cold --execution eager --repeat 50 \
      --csv-out $OUT/smoe_${arm}_$tk.csv > /dev/null 2>&1; echo "smoe $arm $tk rc=$?"
  done
  $BINS/x_add_$arm    --production-only --repeat 50 --csv-out $OUT/add_${arm}.csv  > /dev/null 2>&1; echo "add $arm rc=$?"
  $BINS/x_swig_$arm   --production-only --repeat 50 --csv-out $OUT/swig_${arm}.csv > /dev/null 2>&1; echo "swig $arm rc=$?"
  $BINS/x_lin_$arm    --suite all --repeat 50 --csv-out $OUT/lin_${arm}.csv        > /dev/null 2>&1; echo "lin $arm rc=$?"
  $BINS/x_gdnip_$arm  --repeat 50 --csv-out $OUT/gdnip_${arm}.csv                  > /dev/null 2>&1; echo "gdnip $arm rc=$?"
  $BINS/x_attnip_$arm --repeat 50 --csv-out $OUT/attnip_${arm}.csv                 > /dev/null 2>&1; echo "attnip $arm rc=$?"
  $BINS/x_pair_$arm   --tokens 64,128,192,193,256,384,512,768,1024 --repeat 50 > $OUT/pair_${arm}.txt 2>&1
  echo "pair $arm rc=$?"
done

echo "########## end to end, arms alternating in every round ##########"
m35=/root/models/qwen3_6_35b_a3b.ninfer
m27=/root/models/qwen3_6_27b.ninfer
run() {
  local l=$1 r=$2 b=$3 art=$4 p=$5 ck=$6 tag=$7
  local base=$OUT/${tag}_${l}_r${r}
  $b "$art" --messages "$p" --max-new 32 --greedy --no-thinking \
     --max-context 131072 --prefill-chunk "$ck" --kv-dtype bf16 > "$base.txt" 2> "$base.err"
  echo "$tag $l r$r prefill=$(grep -oE 'prefill speed +[0-9.]+' "$base.err" | grep -oE '[0-9.]+$') decode=$(grep -oE 'decode speed +[0-9.]+' "$base.err" | grep -oE '[0-9.]+$')"
}
pair_run() {   # round artifact prompt chunk tag
  local r=$1 art=$2 p=$3 ck=$4 tag=$5
  if [ $((r % 2)) -eq 1 ]; then
    run MST "$r" $BINS/x_cli_mst "$art" "$p" "$ck" "$tag"
    run W8  "$r" $BINS/x_cli_w8  "$art" "$p" "$ck" "$tag"
  else
    run W8  "$r" $BINS/x_cli_w8  "$art" "$p" "$ck" "$tag"
    run MST "$r" $BINS/x_cli_mst "$art" "$p" "$ck" "$tag"
  fi
}
for r in 1 2 3 4; do
  pair_run "$r" "$m35" /root/qual/fixtures/niah_4k.json  8192 "c8192_niah_4k"
  pair_run "$r" "$m35" /root/qual/fixtures/niah_16k.json 8192 "c8192_niah_16k"
  pair_run "$r" "$m35" /root/qual/fixtures/niah_32k.json 8192 "c8192_niah_32k"
  pair_run "$r" "$m35" /root/qual/fixtures/niah_16k.json 1024 "c1024_niah_16k"
  pair_run "$r" "$m27" /root/qual/fixtures/niah_16k.json 8192 "d27"
done

echo "########## greedy identity ##########"
for f in $(cd $OUT && ls *_MST_*.txt); do
  g=${f/_MST_/_W8_}
  [ -f "$OUT/$g" ] || continue
  if cmp -s "$OUT/$f" "$OUT/$g"; then echo "IDENTICAL ${g}"; else echo "DIFFERS   ${g}"; fi
done

echo "########## sass and resources ##########"
/usr/local/cuda/bin/cuobjdump -sass $BINS/x_cli_mst > /root/x_a.txt 2>/dev/null
/usr/local/cuda/bin/cuobjdump -sass $BINS/x_cli_w8  > /root/x_b.txt 2>/dev/null
python3 /root/sass_diff.py /root/x_a.txt /root/x_b.txt > $OUT/sass_w8.txt 2>&1
python3 /root/resu2.py $BINS/x_cli_mst $BINS/x_cli_w8 > $OUT/resources_w8.txt 2>&1
rm -f /root/x_a.txt /root/x_b.txt
head -3 $OUT/sass_w8.txt
echo W8_ARMS_DONE
