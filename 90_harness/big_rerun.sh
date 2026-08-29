#!/bin/bash
# The sixteen production-extent points were a single audit pass. Two more, so the headline
# figure of the submission is a repeated measurement like everything else it sits beside.
set -u
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/root/bins; OUT=/root/qual/gAUD
for p in 2 3; do
  for arm in mst w8; do
    for nk in "12288 2048" "9216 2048" "2048 4096" "2048 16384"; do
      set -- $nk
      for t in 1024 2048 4096 8192; do
        $B/x_lin_$arm --qtype W8 --n $1 --k $2 --t $t --repeat 50 \
          --csv-out $OUT/big${p}_${arm}_$1_$2_$t.csv > /dev/null 2>&1 || \
        $B/x_lin_$arm --qtype W8 --n $1 --k $2 --sweep $t:$t --repeat 50 \
          --csv-out $OUT/big${p}_${arm}_$1_$2_$t.csv > /dev/null 2>&1
      done
    done
    echo "pass $p $arm done"
  done
done
echo BIG_RERUN_DONE
