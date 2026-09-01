#!/usr/bin/env python3
"""KANDIDAT M-3: warp tile 2x2 instead of 4x1 in the routed gate/up GEMM.

Today a warp owns all four row tiles and one column tile, so per ki it issues four ldmatrix.x4
on A and one x2 on B to feed four mma: five shared reads per four tensor instructions. The A
fragment is not reused at all, and all eight warps read the same A tile in full.

With a 2x2 layout a warp owns two row tiles and two column tiles: two A reads, two B reads, four
mma. That is four shared reads per four tensor instructions, a 20% cut, at zero shared-memory and
zero occupancy cost.

The row pair is {rp, rp + 2}: a gate tile and its matching up tile, so the SiLU gate still runs
inside one warp with no exchange through shared memory.

Per output element the k accumulation order is unchanged and one warp still computes it alone,
so the change must be bit-exact.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")


def sub(old, new, what):
    global t
    if t.count(old) != 1:
        raise SystemExit("anchor %s count %d" % (what, t.count(old)))
    t = t.replace(old, new)


sub("""    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    constexpr int WarpNT        = WarpCols / 8;
    static_assert(ExpertBN % ExpertWarps == 0 && WarpCols % 8 == 0);""",
    """    // A 2x2 warp layout: half the warps hold the upper row tile pair, half the lower, and each
    // takes twice as many columns. A warp's row pair is {rp, rp + 2}, a gate tile and its matching
    // up tile, so the epilogue gate stays inside the warp.
    constexpr int ColGroups     = ExpertWarps / 2;
    constexpr int WarpCols      = ExpertBN / ColGroups;
    constexpr int WarpNT        = WarpCols / 8;
    static_assert(ExpertWarps % 2 == 0 && ExpertBN % ColGroups == 0 && WarpCols % 8 == 0);""",
    "constants")

sub("""    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);""",
    """    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    const int row_pair       = warp / ColGroups;
    const int col_group      = warp - row_pair * ColGroups;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);""",
    "warp indices")

sub("        float acc[4][WarpNT][4] = {};",
    "        float acc[2][WarpNT][4] = {};",
    "acc decl")

sub("""            if (warp * WarpCols < cols) {
                float partial[4][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = warp * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;""",
    """            if (col_group * WarpCols < cols) {
                float partial[2][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[2][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mh = 0; mh < 2; ++mh) {
                        const int row = (row_pair + mh * 2) * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mh][0], af[mh][1], af[mh][2], af[mh][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
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
                    }
                }
#pragma unroll
                for (int mh = 0; mh < 2; ++mh) {
                    const int mi   = row_pair + mh * 2;
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;""",
    "mma loop")

sub("""#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }""",
    """#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mh][ni][0] = fmaf(partial[mh][ni][0], scale0, acc[mh][ni][0]);
                        acc[mh][ni][1] = fmaf(partial[mh][ni][1], scale0, acc[mh][ni][1]);
                        acc[mh][ni][2] = fmaf(partial[mh][ni][2], scale1, acc[mh][ni][2]);
                        acc[mh][ni][3] = fmaf(partial[mh][ni][3], scale1, acc[mh][ni][3]);
                    }""",
    "scale fold")

sub("""        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;
                const int row1 = row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int col0       = begin + column_base + warp * WarpCols + ni * 8 + 2 * lid;
                    const int col1       = col0 + 1;
                    const int local_col0 = warp * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][0]) * acc[mi + 2][ni][0]);
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][2]) * acc[mi + 2][ni][2]);
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][1]) * acc[mi + 2][ni][1]);
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][3]) * acc[mi + 2][ni][3]);
                    }
                }
            }
        }""",
    """        if (col_group * WarpCols < cols) {
            // The warp holds the gate tile in acc[0] and its matching up tile in acc[1], so the
            // gate is applied right here without an exchange through shared memory.
            const int row0 = logical0 + row_pair * 16 + gid;
            const int row1 = row0 + 8;
#pragma unroll
            for (int ni = 0; ni < WarpNT; ++ni) {
                const int local_col0 = col_group * WarpCols + ni * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                const int col0       = begin + column_base + local_col0;
                const int col1       = col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[0][ni][0]) * acc[1][ni][0]);
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[0][ni][2]) * acc[1][ni][2]);
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[0][ni][1]) * acc[1][ni][1]);
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[0][ni][3]) * acc[1][ni][3]);
                }
            }
        }""",
    "epilogue")

F.write_text(t, encoding="utf-8")
print("warp tile 2x2 applied to q4 gate/up")
