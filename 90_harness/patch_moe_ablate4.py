#!/usr/bin/env python3
"""Круг 3: чем ядро занято, если ни счёт, ни трафик его не держат.

Первые два круга дали по 1–6% на каждую фазу, а сумма опознанного — 23%. Такая картина
означает не «узкое место где-то ещё», а латентность: время уходит на ожидание, и снятие любой
одной работы почти не укорачивает цепочку. Проверяется двумя абляциями.

  nowait -- снят cp.async.wait_group: конвейер больше не ждёт прихода данных
  sync   -- сняты два внутренних __syncthreads: остаётся только барьер перед перезаписью стадии

Обе дают неверный результат намеренно.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")

if MODE == "nowait":
    old = """            cp_wait<kExpertStages - 1>();
            __syncthreads();"""
    new = """            // ABLATION: ожидание прихода стадии снято, барьер оставлен.
            __syncthreads();"""
elif MODE == "sync":
    old = """            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight("""
    new = """            cp_wait<kExpertStages - 1>();
            // ABLATION: барьер перед распаковкой снят.
            decode_weight("""
else:
    raise SystemExit("mode must be nowait|sync")

n = t.count(old)
if n == 0:
    raise SystemExit("anchor for %s not found" % MODE)
t = t.replace(old, new)

if MODE == "sync":
    old2 = """            decode_weight(stage);
            __syncthreads();"""
    old3 = """            decode_weight(stage, kt);
            __syncthreads();"""
    m = t.count(old2) + t.count(old3)
    if m == 0:
        raise SystemExit("second sync anchor not found")
    t = t.replace(old2, "            decode_weight(stage);  // ABLATION: барьер после распаковки снят.")
    t = t.replace(old3, "            decode_weight(stage, kt);  // ABLATION: барьер после распаковки снят.")
    n += m

F.write_text(t, encoding="utf-8")
print("patched %s in %d places" % (MODE, n))
