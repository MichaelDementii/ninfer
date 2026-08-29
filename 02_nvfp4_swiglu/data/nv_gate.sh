#!/bin/bash
# A gate that can actually discriminate. The 32-token runs produce 40-byte outputs and only two
# distinct texts across every point, so "identical" there means very little - and here it would
# mislead in the dangerous direction, because the fused and materialising epilogues round
# differently by construction and are not required to agree.
#
# At chunk 1024 nothing about the route changes, so those must agree byte for byte.
# At chunk 8192 the route does change; this measures whether, and where, the product diverges.
set -u
export NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=/root/models/qwen3_8_27b_nvfp4.ninfer
M=/root/models/qwen3_8_27b_nvfp4.ninfer
OUT=/root/qual/gNV
B=/root/bins
for ck in 1024 8192; do
  for arm in mst nv; do
    $B/n_cli_$arm "$M" --messages /root/qual/fixtures/niah_32k.json --max-new 512 --greedy \
      --no-thinking --max-context 131072 --prefill-chunk $ck --kv-dtype bf16 \
      > $OUT/gen512_${arm}_$ck.txt 2> $OUT/gen512_${arm}_$ck.err
    echo "gen512 $arm ck=$ck rc=$? bytes=$(stat -c%s $OUT/gen512_${arm}_$ck.txt)"
  done
done
echo "=== verdicts ==="
for ck in 1024 8192; do
  a=$OUT/gen512_mst_$ck.txt; b=$OUT/gen512_nv_$ck.txt
  if cmp -s "$a" "$b"; then
    echo "chunk $ck: IDENTICAL ($(stat -c%s "$a") bytes)"
  else
    echo "chunk $ck: DIFFERS - first difference at byte $(cmp "$a" "$b" 2>&1 | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+')"
    echo "  master bytes=$(stat -c%s "$a")  branch bytes=$(stat -c%s "$b")"
    echo "  common prefix: $(cmp "$a" "$b" 2>/dev/null; python3 -c "
a=open('$a','rb').read(); b=open('$b','rb').read()
n=0
for x,y in zip(a,b):
    if x!=y: break
    n+=1
print('%d of %d bytes' % (n, min(len(a),len(b))))
")"
  fi
done
echo NV_GATE_DONE
