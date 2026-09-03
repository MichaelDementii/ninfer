#!/usr/bin/env bash
# KONTROL HAOTICHESKOY PERESBORKI VYBORKI.
# Na GPQA fp8 dal 161 iz 198 protiv 171 u int8, Maknemar p=0.034. No 18 raznoglasnyh
# par v OBE storony -- eto pohozhe na to, chto ruki prosto tyanut raznye vyborki:
# pri temperature 0.6 vozmushchenie logitov 1e-4 perebrasyvaet token i vsya cepochka
# rassuzhdeniya rashoditsya.
# Vopros, na kotoryy otvechaet etot zapusk: SKOLKO raznoglasnyh par daet TOT ZHE kodek
# s DRUGIM zhrebiem. Esli stolko zhe -- raznica mezhdu kodekami est shum peresborki.
set -u
LOG=/root/exp/kv_seedctl.log
T=/root/ninfer_d4
D=/root/exp/kv
: > "$LOG"
cd $T || exit 1
export PYTHONPATH=$T/eval

echo "[$(date -u +%H:%M:%S)] zhdem KV_NIAHCTL_DONE" >> "$LOG"
for i in $(seq 1 3000); do grep -q KV_NIAHCTL_DONE /root/exp/kv_niahctl.log 2>/dev/null && break; sleep 30; done
grep -q KV_NIAHCTL_DONE /root/exp/kv_niahctl.log 2>/dev/null || { echo "  ne dozhdalis" >> "$LOG"; exit 0; }

# Konfig, otlichayushchiysya OT ISHODNOGO ROVNO ODNIM CHISLOM -- semenem.
sed 's/seed: 42/seed: 4243/' $D/cfg/text35.yaml > $D/cfg/text35_seed4243.yaml
echo "  konfig s semenem 4243 sozdan, otlichiy ot ishodnogo: $(diff $D/cfg/text35.yaml $D/cfg/text35_seed4243.yaml | grep -c '^[<>]') strok" >> "$LOG"

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

run_arm() { # kv tag
  local kv=$1 tag=$2
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
  $T/eval/.venv/bin/python -m ninfer_eval run --config $D/cfg/text35_seed4243.yaml --suite gpqa \
     > $D/srv/${tag}.eval.log 2>&1
  local rd=$(ls -t $T/eval/runs | head -1)
  [ -f "$T/eval/runs/$rd/summary.md" ] && grep -E "^\| [a-z]" "$T/eval/runs/$rd/summary.md" | sed "s/^/    $tag /" >> "$LOG"
  echo "$tag $rd" >> $D/runmap.txt
  stop_server
}

run_arm int8 "gpqaB_int8"
run_arm fp8  "gpqaB_fp8"

echo "KV_SEEDCTL_DONE" >> "$LOG"
