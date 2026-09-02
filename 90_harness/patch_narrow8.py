#!/usr/bin/env python3
"""Uzkiy plan na vosem varpov: <4,32> -> <8,32>.

Pod ishodnoy raskladkoy 4x1 eto NEVYRAZIMO: WarpCols = ExpertBN / ExpertWarps = 32 / 8 = 4,
i static_assert(WarpCols % 8 == 0) ne prohodit. Pod raskladkoy 2x2 iz paketa 16
ColGroups = 4, WarpCols = 8, WarpNT = 1 -- plitka 2x1, zakonnaya.

Smysl: uzkiy plan otygryvaet nabivku, no platit zanyatostyu (chetyre varpa vmesto vosmi).
Vosem varpov pri BN 32 platyat otnosheniem chteniy (1.5 na mma protiv 1.0 u 2x2 pri BN 64),
no zanyatost sohranyayut. Chto peretyanet -- reshaet zamer.

Primenyat TOLKO poverh patch_warp2x2_v2.py i patch_warp2x2_down.py.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")

SUBS = [
    ("sparse_moe_prefill_q4_gate_up_kernel<4, 32>", "sparse_moe_prefill_q4_gate_up_kernel<8, 32>", 1),
    ("sparse_moe_prefill_qx_down_kernel<Q5DownMma, 4, 32>", "sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 32>", 1),
    ("sparse_moe_prefill_qx_down_kernel<Q6DownMma, 4, 32>", "sparse_moe_prefill_qx_down_kernel<Q6DownMma, 8, 32>", 1),
    ("<<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>", "<<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>", 3),
]
for old, new, want in SUBS:
    got = t.count(old)
    if got != want:
        raise SystemExit("anchor %r: found %d, expected %d" % (old[:48], got, want))
    t = t.replace(old, new)

F.write_text(t, encoding="utf-8")
print("uzkiy plan pereveden na vosem varpov")
