#!/usr/bin/env python3
"""KANDIDAT M-3b: the 2x2 warp tile with the column guard back at eight-column granularity.

Measurement showed why plain 2x2 loses on real routing. With four column tiles per warp the guard
`col_group * WarpCols < cols` skips work in sixteen-column steps, while the 4x1 layout skips in
eight. Expert tiles are only partly filled, so the coarser step makes more warps do work that is
thrown away: under `--distribution same`, where every tile is full, 2x2 wins by 4.8%; under
`trace-like` it loses by 2.7%.

The fix keeps the A-fragment reuse and restores the eight-column step: the guard moves inside the
column loop, so a warp runs exactly the column tiles that carry real columns. Skipped tiles leave
their `partial` at zero, the scale fold then adds zero, and the epilogue already guards each
column, so the result is unchanged.

Applied on top of patch_warp2x2.py.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")

OLD = """#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = col_group * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mh = 0; mh < 2; ++mh) {
                            mma_bf16(partial[mh][ni][0], partial[mh][ni][1], partial[mh][ni][2],
                                     partial[mh][ni][3], af[mh][0], af[mh][1], af[mh][2], af[mh][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }"""

NEW = """#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        // The guard stays at eight columns, the width of one mma tile: a warp
                        // owning sixteen columns must not pay for the upper eight when the
                        // expert's assignment count stops inside them.
                        if (col_group * WarpCols + ni * 8 >= cols) { continue; }
                        const int brow = col_group * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mh = 0; mh < 2; ++mh) {
                            mma_bf16(partial[mh][ni][0], partial[mh][ni][1], partial[mh][ni][2],
                                     partial[mh][ni][3], af[mh][0], af[mh][1], af[mh][2], af[mh][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }"""

if t.count(OLD) != 1:
    raise SystemExit("anchor count %d (apply patch_warp2x2.py first)" % t.count(OLD))
F.write_text(t.replace(OLD, NEW), encoding="utf-8")
print("2x2 column guard narrowed to eight columns")
