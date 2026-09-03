#!/usr/bin/env bash
# KONTROL CHUVSTVITELNOSTI "igolki".
# Esli nvfp4 (vdesyatero huzhe fp8 po perplexii) i k8v4 (vchetvero huzhe) tozhe
# dayut idealnoe izvlechenie, znachit pribor ne razlichaet kodeki voobshche, i nol
# promahov u fp8 nichego ne dokazyvaet. Etot zapusk otvechaet na etot vopros.
set -u
LOG=/root/exp/kv_niahctl.log
T=/root/ninfer_d4
D=/root/exp/kv
: > "$LOG"
cd $T || exit 1
export PYTHONPATH=$T/eval

echo "[$(date -u +%H:%M:%S)] zhdem KV_AFTER_DONE" >> "$LOG"
for i in $(seq 1 3000); do grep -q KV_AFTER_DONE /root/exp/kv_after.log 2>/dev/null && break; sleep 30; done
grep -q KV_AFTER_DONE /root/exp/kv_after.log 2>/dev/null || { echo "  ne dozhdalis, vyhodim" >> "$LOG"; exit 0; }

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

run_arm() { # kv ctx suite tag
  local kv=$1 ctx=$2 suite=$3 tag=$4
  curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { echo "  port zanyat, propusk $tag" >> "$LOG"; return 1; }
  /root/bins/CAND/ninfer-serve /root/models/qwen3_6_35b_a3b.ninfer --host 127.0.0.1 --port 18080 \
    --model-id qwen3.6-35b-a3b --max-context "$ctx" --kv-capacity auto --max-concurrency 1 \
    --max-pending-requests 1 --pending-timeout-ms 86400000 --prefill-chunk 1024 --kv-dtype "$kv" \
    > $D/srv/${tag}.server.log 2>&1 &
  SRV_PID=$!
  local ok=0
  for i in $(seq 1 300); do
    curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SRV_PID 2>/dev/null || break
    sleep 1
  done
  [ $ok -eq 1 ] || { echo "  server ne podnyalsya: $tag" >> "$LOG"; SRV_PID=""; return 1; }
  echo "[$(date -u +%H:%M:%S)] $tag" >> "$LOG"
  $T/eval/.venv/bin/python -m ninfer_eval run --config /root/exp/kv/cfg/niah35.yaml --suite "$suite" \
     > $D/srv/${tag}.eval.log 2>&1
  local rd=$(ls -t $T/eval/runs | head -1)
  [ -f "$T/eval/runs/$rd/summary.md" ] && grep -E "^\| [a-z]" "$T/eval/runs/$rd/summary.md" | sed "s/^/    $tag /" >> "$LOG"
  echo "$tag $rd" >> $D/runmap.txt
  stop_server
}

for KV in nvfp4 k8v4; do
  run_arm $KV 40960  standard  "standard_${KV}"
  run_arm $KV 139264 long_128k "long_128k_${KV}"
done

echo "KV_NIAHCTL_DONE" >> "$LOG"
