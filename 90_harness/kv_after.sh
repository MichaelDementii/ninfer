#!/usr/bin/env bash
# Posle nochi: to, chto ne vlezlo v osnovnoy skript.
set -u
LOG=/root/exp/kv_after.log
: > "$LOG"
echo "[$(date -u +%H:%M:%S)] zhdem KV_NIGHT_DONE" >> "$LOG"
for i in $(seq 1 3000); do grep -q KV_NIGHT_DONE /root/exp/kv_night.log 2>/dev/null && break; sleep 30; done
if ! grep -q KV_NIGHT_DONE /root/exp/kv_night.log 2>/dev/null; then
  echo "  noch ne zavershilas za otvedennoe vremya, vyhodim" >> "$LOG"; exit 0
fi
echo "[$(date -u +%H:%M:%S)] noch zavershena" >> "$LOG"

echo "[$(date -u +%H:%M:%S)] 1) testy na realnom artefakte 35B" >> "$LOG"
/root/exp/kv_tests35.sh
tail -6 /root/exp/kv_tests35.log | sed 's/^/    /' >> "$LOG"

echo "[$(date -u +%H:%M:%S)] 2) perplexiya na sklennom dokumente 260k pri kontekste 262144" >> "$LOG"
/root/exp/kv_ppl_long.sh
grep -E "kontekst|overall|rc=" /root/exp/kv_ppl_long.log | sed 's/^/    /' >> "$LOG"

echo "KV_AFTER_DONE" >> "$LOG"
