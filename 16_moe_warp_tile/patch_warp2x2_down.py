#!/usr/bin/env python3
"""KANDIDAT M-3b: the same 2x2 warp tile in the routed down-projection GEMM.

qx_down has the identical 4x1 layout: a warp owns all four row tiles and the eight columns at
warp * 8, so per ki it issues four ldmatrix.x4 on A and one x2 on B for four mma. Four warps'
worth of column tiles and two row tiles per warp turn that into two A reads plus two B reads for
four mma.

Unlike gate/up there is no gate/up pairing in the epilogue here, so the row pair is simply
{2 * rp, 2 * rp + 1}.

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


# 1. layout constants and warp indices, inside qx_down only (anchored on its GroupsPerRow)
sub("""    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kIntermediate / 64;""",
    """    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kIntermediate / 64;
    // A 2x2 warp tile: two row tiles and two column tiles per warp instead of four and one, so
    // each A fragment feeds two mma instead of one.
    constexpr int ColGroups     = ExpertWarps / 2;
    constexpr int WarpCols      = ExpertBN / ColGroups;
    constexpr int WarpNT        = WarpCols / 8;
    static_assert(ExpertWarps % 2 == 0 && ExpertBN % ColGroups == 0 && WarpCols % 8 == 0);""",
    "down constants")

sub("""    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;""",
    """    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    const int row_pair       = warp / ColGroups;
    const int col_group      = warp - row_pair * ColGroups;
    constexpr int row_blocks = kHidden / kExpertBM;""",
    "down warp indices")

sub("        float acc[4][4]       = {};",
    "        float acc[2][WarpNT][4] = {};",
    "down acc decl")

# 2. main loop
sub("""            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }""",
    """            if (col_group * WarpCols < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[2][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mh = 0; mh < 2; ++mh) {
                        const int row = (row_pair * 2 + mh) * 16 + a_rowoff;
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
                            mma_bf16(acc[mh][ni][0], acc[mh][ni][1], acc[mh][ni][2],
                                     acc[mh][ni][3], af[mh][0], af[mh][1], af[mh][2], af[mh][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }
                }
            }""",
    "down mma loop")

# 3. epilogue
sub("""        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }""",
    """        if (col_group * WarpCols < cols) {
#pragma unroll
            for (int mh = 0; mh < 2; ++mh) {
                const int output_row0 = row0 + (row_pair * 2 + mh) * 16 + gid;
                const int output_row1 = output_row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int local_col0 = col_group * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    const int col0       = begin + column_base + local_col0;
                    const int col1       = col0 + 1;
                    if (local_col0 < cols) {
                        output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                            __float2bfloat16_rn(acc[mh][ni][0]);
                        output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                            __float2bfloat16_rn(acc[mh][ni][2]);
                    }
                    if (local_col1 < cols) {
                        output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                            __float2bfloat16_rn(acc[mh][ni][1]);
                        output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                            __float2bfloat16_rn(acc[mh][ni][3]);
                    }
                }
            }
        }""",
    "down epilogue")

F.write_text(t, encoding="utf-8")
print("warp tile 2x2 applied to qx down")
