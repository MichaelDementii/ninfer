#!/usr/bin/env bash
# Sobstvennye chislennye testy proekta na REALNOM artefakte 35B.
# Nochnoy skript zapuskaet ih bez NINFER_QWEN3_6_35B_A3B_WEIGHTS, poetomu oni tam
# propuskayutsya. Zdes' oni proganyayutsya kak polozheno.
set -u
LOG=/root/exp/kv_tests35.log
: > "$LOG"
export NINFER_QWEN3_6_35B_A3B_WEIGHTS=/root/models/qwen3_6_35b_a3b.ninfer
cd /root/ninfer_d4/build || exit 1
echo "[$(date -u +%H:%M:%S)] ctest na realnom artefakte 35B" >> "$LOG"
ctest --output-on-failure -R "35b_a3b" > /root/exp/kv/ctest35.log 2>&1
echo "  rc=$? -- $(grep -E '^[0-9]+% tests passed' /root/exp/kv/ctest35.log | tail -1)" >> "$LOG"
grep -E "Passed|Failed|\*\*\*|Test #" /root/exp/kv/ctest35.log | tail -20 | sed 's/^/    /' >> "$LOG"
echo "KV_TESTS35_DONE" >> "$LOG"
