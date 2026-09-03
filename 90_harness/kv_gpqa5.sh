#!/usr/bin/env bash
# Dovodka GPQA do pyati semyan na ruku.
# Pri treh tochkah: int8 168.33+-2.52, fp8 163.33+-2.08, p=0.059 -- na grani.
# Pri pyati tochkah, esli razryv realen (~5 zadach pri SD 2.3), t vyrastet do ~3.4
# i vopros zakroetsya v odnu ili druguyu storonu.
set -u
LOG=/root/exp/kv_gpqa5.log
T=/root/ninfer_d4
D=/root/exp/kv
: > "$LOG"
cd $T || exit 1
export PYTHONPATH=$T/eval

for i in $(seq 1 200); do grep -q KV_FILL_DONE /root/exp/kv_fill.log 2>/dev/null && break; sleep 20; done

SRV_PID=""
stop_server() {
  [ -n "$SRV_PID" ] || return 0
  kill "$SRV_PID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""
  for i in $(seq 1 120); do curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 || break; sleep 1; done
  sleep 2
}
trap 'stop_server' EXIT

run_arm() { # kv cfg tag
  local kv=$1 cfg=$2 tag=$3
  curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { echo "  port zanyat, propusk $tag" >> "$LOG"; return 1; }
  /root/bins/CAND/ninfer-serve /root/models/qwen3_6_35b_a3b.ninfer --host 127.0.0.1 --port 18080 \
    --model-id qwen3.6-35b-a3b --max-context 131072 --kv-capacity auto --max-concurrency 4 \
    --max-pending-requests 4 --pending-timeout-ms 86400000 --prefill-chunk 1024 --kv-dtype "$kv" \
    --spec mtp --draft-tokens 3 --lm-head-draft > $D/srv/${tag}.server.log 2>&1 &
  SRV_PID=$!
  local ok=0
  for i in $(seq 1 300); do
    curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SRV_PID 2>/dev/null || break
    sleep 1
  done
  [ $ok -eq 1 ] || { echo "  server ne podnyalsya: $tag" >> "$LOG"; SRV_PID=""; return 1; }
  echo "[$(date -u +%H:%M:%S)] $tag" >> "$LOG"
  $T/eval/.venv/bin/python -m ninfer_eval run --config "$cfg" --suite gpqa > $D/srv/${tag}.eval.log 2>&1
  local rd=$(ls -t $T/eval/runs | head -1)
  [ -f "$T/eval/runs/$rd/summary.md" ] && grep -E "^\| [a-z]" "$T/eval/runs/$rd/summary.md" | sed "s/^/    $tag /" >> "$LOG"
  echo "$tag $rd" >> $D/runmap.txt
  stop_server
}

for S in 31337 90210; do
  sed "s/seed: 42/seed: $S/" $D/cfg/text35.yaml > $D/cfg/text35_seed$S.yaml
  run_arm int8 "$D/cfg/text35_seed$S.yaml" "gpqaS${S}_int8"
  run_arm fp8  "$D/cfg/text35_seed$S.yaml" "gpqaS${S}_fp8"
done

echo "KV_GPQA5_DONE" >> "$LOG"
