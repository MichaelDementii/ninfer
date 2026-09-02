#!/usr/bin/env bash
# Dobor k zameru skorosti:
#  A) dlinnyy kontekst 65536 i 131072 -- tam, gde dolya KV maksimalna;
#  B) proizvodstvennyy rezhim avtora: mtp3 + --lm-head-draft. Gipoteza: bez flaga
#     kazhdyy raund lishniy raz chitaet bolshuyu Q6-golovu (397 MB protiv 143 MB),
#     i eto razbavlyaet dolyu KV, zanizhaya vyigrysh kodeka.
set -u
LOG=/root/exp/kv_speed2.log
D=/root/exp/kv
: > "$LOG"
B=/root/bins/CAND/ninfer_bench
M=/root/models/qwen3_6_35b_a3b.ninfer

row() {
  [ -s "$1" ] || { echo "  $2 NET_DANNYH" >> "$LOG"; return; }
  tail -n +2 "$1" | awk -F, -v l="$2" '{
    ms = ($23>0) ? 1000*$35*$26/$23 : 1000*$35/$4;
    printf "  %-26s pp=%-7s pref_tok_s=%9.2f dec_tok_s=%8.3f ms_round=%8.4f acc=%s kv_B=%s\n",
           l, $3, $27, $29, ms, $25, $13 }' >> "$LOG"
}

echo "[$(date -u +%H:%M:%S)] ##### A) dlinnyy kontekst, mtp=0 #####" >> "$LOG"
for leg in a b; do
  for KV in bf16 int8 fp8 nvfp4 k8v4; do
    f=$D/L0_${KV}_$leg.csv
    $B --weights $M -pg "65536,128" --kv-dtype $KV --prefill-chunk 1024 \
       --mtp-draft-tokens 0 -r 3 --warmup 1 -o csv --output-file $f > $D/L0_${KV}_$leg.err 2>&1
    row $f "mtp0/$KV/$leg"
  done
done

echo "" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] ##### B) proizvodstvennyy rezhim: mtp3 + lm-head-draft #####" >> "$LOG"
for leg in a b; do
  for KV in bf16 int8 fp8 nvfp4 k8v4; do
    f=$D/P3_${KV}_$leg.csv
    $B --weights $M -pg "1024,128;8192,128;32768,128" --kv-dtype $KV --prefill-chunk 1024 \
       --mtp-draft-tokens 3 --lm-head-draft -r 5 --warmup 2 -o csv --output-file $f > /dev/null 2>&1
    row $f "mtp3d/$KV/$leg"
  done
done

echo "" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] ##### C) 131072, mtp=0, tolko fp8 protiv int8 i bf16 #####" >> "$LOG"
for KV in bf16 int8 fp8; do
  f=$D/X0_${KV}.csv
  $B --weights $M -pg "131072,128" --kv-dtype $KV --prefill-chunk 1024 \
     --mtp-draft-tokens 0 -r 2 --warmup 1 -o csv --output-file $f > $D/X0_${KV}.err 2>&1
  row $f "mtp0/$KV/x"
  [ -s "$f" ] || tail -2 $D/X0_${KV}.err | sed 's/^/    ERR /' >> "$LOG"
done

echo "KV_SPEED2_DONE" >> "$LOG"
