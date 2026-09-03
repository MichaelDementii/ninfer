#!/usr/bin/env bash
# Perplexiya na sklennom dokumente ~260 tys. tokenov pri kontekste 262144.
# Potoki korpusa po 65 tys. do etogo rezhima ne dostayut, a imenno tam oshibka
# kodeka kopitsya silnee vsego: ves' kesh KV zhivet v vybrannom kodeke srazu.
set -u
LOG=/root/exp/kv_ppl_long.log
D=/root/exp/kv/ppl_long
mkdir -p $D
: > "$LOG"
cd /root/ninfer_d4 || exit 1
B=/root/bins/CAND/ninfer-perplexity
M=/root/models/qwen3_6_35b_a3b.ninfer
C=eval/corpora/perplexity-1m/data

# Sklejka: po odnomu potoku iz kazhdogo domena, chtoby dlinnyy dokument ne byl
# odnorodnym. ~4 x 65 tys. = ~260 tys. tokenov.
BIG=$D/long260k.txt
if [ ! -s "$BIG" ]; then
  cat $C/wikitext/00.txt $C/pg19/00.txt $C/zhwiki/00.txt $C/ninfer/00.txt > "$BIG"
fi
echo "[$(date -u +%H:%M:%S)] sklennyy dokument: $(wc -c < "$BIG") bayt" >> "$LOG"

for KV in bf16 int8 fp8 k8v4 nvfp4; do
  out=$D/$KV; rm -rf "$out"; mkdir -p "$out"
  echo "[$(date -u +%H:%M:%S)] --- $KV, kontekst 262144, shag 131072" >> "$LOG"
  t0=$(date +%s)
  $B $M --text "$BIG" --context 262144 --stride 131072 --kv-dtype $KV --output "$out" \
     > $D/$KV.stdout 2> $D/$KV.stderr
  rc=$?; t1=$(date +%s)
  echo "    rc=$rc za $((t1-t0)) s" >> "$LOG"
  if [ -s $D/$KV.stdout ]; then
    grep -iE "overall|perplex" $D/$KV.stdout | tail -4 | sed 's/^/    /' >> "$LOG"
  else
    tail -4 $D/$KV.stderr | sed 's/^/    ERR /' >> "$LOG"
  fi
done
echo "KV_PPL_LONG_DONE" >> "$LOG"
