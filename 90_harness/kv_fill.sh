#!/usr/bin/env bash
# Dozamer: bf16 na GPQA i IFBench. U umolchaniya dvizhka est AIME, no net etih dvuh,
# a sravnenie imenno s umolchaniem -- samoe cennoe dlya produkta.
# Plyus tretye semya na GPQA dlya oboih kodekov: s tremya tochkami na ruku
# sravnenie srednih poluchaet realnuyu moshchnost.
set -u
LOG=/root/exp/kv_fill.log
T=/root/ninfer_d4
D=/root/exp/kv
: > "$LOG"
cd $T || exit 1
export PYTHONPATH=$T/eval

for i in $(seq 1 200); do grep -q KV_SEEDCTL_DONE /root/exp/kv_seedctl.log 2>/dev/null && break; sleep 30; done

sed 's/seed: 42/seed: 777/' $D/cfg/text35.yaml > $D/cfg/text35_seed777.yaml

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

run_arm() { # kv cfg suite tag
  local kv=$1 cfg=$2 suite=$3 tag=$4
  curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { echo "  port zanyat, propusk $tag" >> "$LOG"; return 1; }
  /root/bins/CAND/ninfer-serve /root/models/qwen3_6_35b_a3b.ninfer --host 127.0.0.1 --port 18080 \
    --model-id qwen3.6-35b-a3b --max-context 131072 --kv-capacity auto --max-concurrency 4 \
    --max-pending-requests 4 --pending-timeout-ms 86400000 --prefill-chunk 1024 --kv-dtype "$kv" \
    --spec mtp --draft-tokens 3 --lm-head-draft \
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
  $T/eval/.venv/bin/python -m ninfer_eval run --config "$cfg" --suite "$suite" > $D/srv/${tag}.eval.log 2>&1
  local rd=$(ls -t $T/eval/runs | head -1)
  [ -f "$T/eval/runs/$rd/summary.md" ] && grep -E "^\| [a-z]" "$T/eval/runs/$rd/summary.md" | sed "s/^/    $tag /" >> "$LOG"
  echo "$tag $rd" >> $D/runmap.txt
  stop_server
}

C=$D/cfg/text35.yaml
C7=$D/cfg/text35_seed777.yaml

run_arm bf16 "$C"  gpqa    "gpqa_bf16"
run_arm bf16 "$C"  ifbench "ifbench_bf16"
run_arm int8 "$C7" gpqa    "gpqaC_int8"
run_arm fp8  "$C7" gpqa    "gpqaC_fp8"

echo "KV_FILL_DONE" >> "$LOG"
