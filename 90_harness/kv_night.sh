#!/usr/bin/env bash
# Nochnaya kampaniya po kodeku KV fp8.
# Odna dvoichnaya CAND na vsyo. Menyaetsya rovno odin flag servera: --kv-dtype.
# Poryadok -- po chuvstvitelnosti pribora na chas raboty:
#   perplexiya -> igolka v stoge -> AIME -> IFBench -> GPQA.
set -u
LOG=/root/exp/kv_night.log
T=/root/ninfer_d4
BIN=/root/bins/CAND
M=/root/models/qwen3_6_35b_a3b.ninfer
D=/root/exp/kv
mkdir -p $D/srv
: > "$LOG"
cd $T || exit 1
export PYTHONPATH=$T/eval

say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

SRV_PID=""
stop_server() {
  [ -n "$SRV_PID" ] || return 0
  kill "$SRV_PID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV_PID" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
  SRV_PID=""
  # Port dolzhen realno osvoboditsya. Inache proverka zdorovya sleduyushchego etapa
  # uvidit STARYY server i ruka izmeritsya chuzhim kodekom.
  for i in $(seq 1 120); do
    curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1 || break
    sleep 1
  done
  if curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
    say "    VNIMANIE: port 18080 vsyo eshche otvechaet posle ostanovki"
  fi
  sleep 2
}
trap 'stop_server' EXIT

# start_server <kv> <ctx> <conc> <spec:yes|no> <tag>
start_server() {
  local kv=$1 ctx=$2 conc=$3 spec=$4 tag=$5
  local slog=$D/srv/${tag}.server.log
  local rlog=$D/srv/${tag}.requests.jsonl
  if curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
    say "    OTKAZ: port 18080 zanyat chuzhim serverom, etap propushchen ($tag)"
    return 1
  fi
  local -a extra=()
  if [ "$spec" = "yes" ]; then extra=(--spec mtp --draft-tokens 3 --lm-head-draft); fi
  $BIN/ninfer-serve $M --host 127.0.0.1 --port 18080 --model-id qwen3.6-35b-a3b \
      --max-context "$ctx" --kv-capacity auto \
      --max-concurrency "$conc" --max-pending-requests "$conc" \
      --pending-timeout-ms 86400000 --prefill-chunk 1024 \
      --kv-dtype "$kv" "${extra[@]}" \
      --request-log-jsonl "$rlog" > "$slog" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 300); do
    if curl --fail --silent --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
      say "    server podnyat: kv=$kv ctx=$ctx conc=$conc spec=$spec"
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      say "    SERVER UPAL do gotovnosti ($tag): $(grep -iE 'error|fail|invalid|outside' $slog | tail -2 | tr '\n' ' ')"
      SRV_PID=""; return 1
    fi
    sleep 1
  done
  say "    SERVER NE PODNYALSYA za 300 s ($tag)"
  stop_server; return 1
}

# evalrun <config> <suite> <tag>
evalrun() {
  local cfg=$1 suite=$2 tag=$3
  local t0=$(date +%s)
  say "    ocenka: $suite [$tag]"
  $T/eval/.venv/bin/python -m ninfer_eval run --config "$cfg" --suite "$suite" \
     > $D/srv/${tag}.eval.log 2>&1
  local rc=$?
  local t1=$(date +%s)
  local rd=$(ls -t $T/eval/runs 2>/dev/null | head -1)
  say "    rc=$rc za $(( (t1-t0)/60 )) min, run=$rd"
  if [ -f "$T/eval/runs/$rd/summary.md" ]; then
    grep -E "^\| [a-z]" "$T/eval/runs/$rd/summary.md" | sed "s/^/      $tag /" >> "$LOG"
    echo "$tag $rd" >> $D/runmap.txt
  else
    tail -4 $D/srv/${tag}.eval.log | sed 's/^/      ERR /' >> "$LOG"
  fi
}

# stage <config> <suite> <ctx> <conc> <spec> <arms...>
stage() {
  local cfg=$1 suite=$2 ctx=$3 conc=$4 spec=$5; shift 5
  say "===== ETAP: $suite (ctx=$ctx conc=$conc spec=$spec) ====="
  for kv in "$@"; do
    if start_server "$kv" "$ctx" "$conc" "$spec" "${suite}_${kv}"; then
      evalrun "$cfg" "$suite" "${suite}_${kv}"
      stop_server
    fi
  done
}


# Razvedka: kakoy max-context voobshche startuet na etoy karte s etim artefaktom.
# Odin otkaz starta inache unes by ves tekstovyy nabor.
CTX_TEXT=0
probe_ctx() {
  for c in 131072 98304 73728; do
    if start_server int8 "$c" 4 yes "probe_$c"; then
      CTX_TEXT=$c; stop_server
      say "   maksimalnyy kontekst pri conc=4: $CTX_TEXT"
      return 0
    fi
  done
  say "   RAZVEDKA NE NASHLA rabochiy kontekst"
  return 1
}

CFG_T=/root/exp/kv/cfg/text35.yaml
CFG_N=/root/exp/kv/cfg/niah35.yaml

say "########## NACHALO. bin=CAND (camp36 + 2x2 + porog 1280) ##########"

say "0) zhdem zaversheniya zamerov skorosti i dymovogo testa"
for i in $(seq 1 400); do grep -q SMOKE_DONE /root/exp/kv_smoke.log 2>/dev/null && break; sleep 15; done
say "   zamer skorosti zavershen"

say "1) PERPLEXIYA po pyati kodekam, dva protokola"
/root/exp/kv_ppl.sh
say "   perplexiya zavershena"

say "1b) sobstvennye chislennye testy proekta po kodekam KV i po realnomu artefaktu 35B"
( cd /root/ninfer_d4/build && ctest --output-on-failure     -R "softmax_attention|kv_cache_append|kv_cache_test|kv_capacity|35b_a3b"     > /root/exp/kv/ctest.log 2>&1 )
say "   ctest rc=$? -- $(grep -E '^[0-9]+% tests passed' /root/exp/kv/ctest.log | tail -1)"
grep -E "Passed|Failed|\*\*\*" /root/exp/kv/ctest.log | tail -20 | sed 's/^/      /' >> "$LOG"

# Igolka v stoge -- bez spekulyacii: maksimalnaya opredelennost, kodek izolirovan.
stage $CFG_N standard  40960 1 no  bf16 int8 fp8
stage $CFG_N long_64k  73728 1 no  bf16 int8 fp8
stage $CFG_N long_128k 139264 1 no bf16 int8 fp8
stage $CFG_N long_260k 262144 1 no bf16 int8 fp8

# Sposobnosti -- polnyy recept avtora (spekulyaciya i chernovaya golova vklyucheny),
# odinakovyy na obeih rukah; menyaetsya tolko kodek.
say "5) razvedka konteksta dlya tekstovogo nabora"
if probe_ctx; then
  # Byudzhety vyvoda podgonyayutsya pod nashedennyy kontekst, chtoby zapros ne otvergli.
  python3 /root/exp/mkcfg.py --ctx "$CTX_TEXT" >> "$LOG" 2>&1
  stage $CFG_T aime_both "$CTX_TEXT" 4 yes int8 fp8 bf16
  stage $CFG_T gpqa      "$CTX_TEXT" 4 yes int8 fp8
  stage $CFG_T ifbench   "$CTX_TEXT" 4 yes int8 fp8
fi

say "########## KV_NIGHT_DONE ##########"
