#!/usr/bin/env bash
# Polnaya matrica: nvfp4 i k8v4 na sposobnostyah.
# U nih uzhe est perplexiya (tri protokola), igolka (dva profilya) i ctest.
# Ne hvatalo AIME, GPQA i IFBench -- zdes' oni dobirayutsya tem zhe receptom
# avtora i tem zhe semenem 42, chto i u bf16/int8/fp8.
set -u
LOG=/root/exp/kv_allcodec.log
T=/root/ninfer_d4
D=/root/exp/kv
: > "$LOG"
cd $T || exit 1
export PYTHONPATH=$T/eval

echo "[$(date -u +%H:%M:%S)] zhdem KV_GPQA5_DONE" >> "$LOG"
for i in $(seq 1 400); do grep -q KV_GPQA5_DONE /root/exp/kv_gpqa5.log 2>/dev/null && break; sleep 20; done

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

run_arm() { # kv suite tag
  local kv=$1 suite=$2 tag=$3
  curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { echo "  port zanyat, propusk $tag" >> "$LOG"; return 1; }
  /root/bins/CAND/ninfer-serve /root/models/qwen3_6_35b_a3b.ninfer --host 127.0.0.1 --port 18080 \
    --model-id qwen3.6-35b-a3b --max-context 131072 --kv-capacity auto --max-concurrency 4 \
    --max-pending-requests 4 --pending-timeout-ms 86400000 --prefill-chunk 1024 --kv-dtype "$kv" \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --request-log-jsonl $D/srv/${tag}.requests.jsonl > $D/srv/${tag}.server.log 2>&1 &
  SRV_PID=$!
  local ok=0
  for i in $(seq 1 300); do
    curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SRV_PID 2>/dev/null || break
    sleep 1
  done
  [ $ok -eq 1 ] || { echo "  server ne podnyalsya: $tag" >> "$LOG"; SRV_PID=""; return 1; }
  echo "[$(date -u +%H:%M:%S)] $tag" >> "$LOG"
  $T/eval/.venv/bin/python -m ninfer_eval run --config $D/cfg/text35.yaml --suite "$suite" \
     > $D/srv/${tag}.eval.log 2>&1
  local rd=$(ls -t $T/eval/runs | head -1)
  [ -f "$T/eval/runs/$rd/summary.md" ] && grep -E "^\| [a-z]" "$T/eval/runs/$rd/summary.md" | sed "s/^/    $tag /" >> "$LOG"
  echo "$tag $rd" >> $D/runmap.txt
  stop_server
}

# Poryadok: snachala deshevoe i na oboih kodekah, chtoby lyubaya ostanovka
# ostavila polnye pary, a ne polovinu.
for KV in nvfp4 k8v4; do run_arm $KV aime_both "aime_both_${KV}"; done
for KV in nvfp4 k8v4; do run_arm $KV gpqa      "gpqa_${KV}";      done
for KV in nvfp4 k8v4; do run_arm $KV ifbench   "ifbench_${KV}";   done

echo "KV_ALLCODEC_DONE" >> "$LOG"
