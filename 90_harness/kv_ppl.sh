#!/usr/bin/env bash
# Perplexiya po kodekam KV. Protokol dlya sravneniya kodekov -- avtorskiy:
# polnyy korpus, --context 65536 --stride 32768 (docs/perplexity.md).
# Vtoroy protokol -- umolchanie 4096/2048, dlya preemstvennosti s proshlym krugom.
set -u
LOG=/root/exp/kv_ppl.log
D=/root/exp/kv/ppl
mkdir -p $D
: > "$LOG"
cd /root/ninfer_d4 || exit 1
B=/root/bins/CAND/ninfer-perplexity
M=/root/models/qwen3_6_35b_a3b.ninfer
C=eval/corpora/perplexity-1m/manifest.json

run() { # ctx stride kv tag
  local ctx=$1 str=$2 kv=$3 tag=$4
  local out=$D/$tag
  rm -rf "$out"; mkdir -p "$out"
  echo "[$(date -u +%H:%M:%S)] --- $tag: ctx=$ctx stride=$str kv=$kv" >> "$LOG"
  local t0=$(date +%s)
  $B $M --corpus $C --context $ctx --stride $str --kv-dtype $kv --output "$out" \
     > $D/$tag.stdout 2> $D/$tag.stderr
  local rc=$?
  local t1=$(date +%s)
  echo "    rc=$rc za $((t1-t0)) s" >> "$LOG"
  if [ -s $D/$tag.stdout ]; then
    grep -iE "overall|perplex|domain|english|chinese|code" $D/$tag.stdout | tail -14 | sed 's/^/    /' >> "$LOG"
  else
    tail -5 $D/$tag.stderr | sed 's/^/    ERR /' >> "$LOG"
  fi
}

echo "[$(date -u +%H:%M:%S)] ===== PROTOKOL AVTORA: polnyy korpus 65536/32768 =====" >> "$LOG"
for KV in bf16 int8 fp8 k8v4 nvfp4; do run 65536 32768 $KV "L_$KV"; done

echo "" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] ===== PROTOKOL PO UMOLCHANIYU: polnyy korpus 4096/2048 =====" >> "$LOG"
for KV in bf16 int8 fp8 k8v4 nvfp4; do run 4096 2048 $KV "S_$KV"; done

echo "KV_PPL_DONE" >> "$LOG"
