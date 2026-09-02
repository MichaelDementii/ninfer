#!/usr/bin/env bash
# Bystryy zamer skorosti po kodekam KV. Odna dvoichnaya CAND, menyaetsya tolko flag.
# Metrika: pri mtp=0 -- decode_output_tok_s (chistaya cena kodeka, priyomka ne meshaet);
#          pri mtp=3 -- ms/raund (real'nyy rezhim, priyomka pokazana ryadom).
set -u
LOG=/root/exp/kv_speed.log
D=/root/exp/kv
mkdir -p $D
: > "$LOG"
B=/root/bins/CAND/ninfer_bench
M=/root/models/qwen3_6_35b_a3b.ninfer

row() { # file label
  [ -s "$1" ] || { echo "  $2 NET_DANNYH" >> "$LOG"; return; }
  tail -n +2 "$1" | awk -F, -v l="$2" '{
    ms = ($23>0) ? 1000*$35*$26/$23 : 1000*$35/$4;
    printf "  %-28s pp=%-6s pref_tok_s=%9.2f pref_s=%8.4f dec_tok_s=%8.3f ms_round=%8.4f acc=%s kv_B=%s\n",
           l, $3, $27, $34, $29, ms, $25, $13 }' >> "$LOG"
}

echo "[$(date -u +%H:%M:%S)] === START. bin=CAND (camp36 + 2x2 + porog 1280) ===" >> "$LOG"

for MTP in 0 3; do
  echo "" >> "$LOG"
  echo "[$(date -u +%H:%M:%S)] ##### mtp-draft-tokens = $MTP #####" >> "$LOG"
  for leg in a b; do
    for KV in bf16 int8 fp8 nvfp4 k8v4; do
      f=$D/sp_${MTP}_${KV}_${leg}.csv
      $B --weights $M -pg "1024,128;8192,128;32768,128" \
         --kv-dtype $KV --prefill-chunk 1024 --mtp-draft-tokens $MTP \
         -r 5 --warmup 2 -o csv --output-file $f > /dev/null 2>&1
      row $f "mtp$MTP/$KV/$leg"
    done
  done
done

echo "" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] ##### dlinnyy kontekst 65536, mtp=0 i mtp=3 #####" >> "$LOG"
for MTP in 0 3; do
  for KV in bf16 int8 fp8 k8v4; do
    f=$D/lg_${MTP}_${KV}.csv
    $B --weights $M -pg "65536,128" --max-context 70000 \
       --kv-dtype $KV --prefill-chunk 1024 --mtp-draft-tokens $MTP \
       -r 3 --warmup 1 -o csv --output-file $f > /dev/null 2>&1
    row $f "L/mtp$MTP/$KV"
  done
done

echo "" >> "$LOG"
echo "[$(date -u +%H:%M:%S)] KV_SPEED_DONE" >> "$LOG"
