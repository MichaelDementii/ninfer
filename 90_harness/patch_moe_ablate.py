#!/usr/bin/env python3
"""Ablations on the routed MoE prefill GEMM, to replace reasoning with measurement.

The kernel runs `decode -> barrier -> mma -> scale-fold -> barrier -> stage` per k-tile, so the
Q4 unpack cannot overlap the tensor work. Each stage below removes exactly one of those phases
while keeping every operand live, so the compiler cannot delete the staging around it.

  mma    -- tensor instruction replaced by a cheap dependency on all six operand registers
  dec    -- Q4 unpack replaced by a cheap function of the staged codes (load and store survive)
  fold   -- per-k-tile scale application (two shuffles + 16 FFMA per warp) dropped
  shfl   -- CANDIDATE, not an ablation: the quad broadcast is a shared-memory broadcast read,
            so __shfl_sync is redundant. Bit-exact.

Results are wrong on purpose for mma/dec/fold; only shfl must stay bit-exact.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
F = ROOT / "src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
t = F.read_text(encoding="utf-8")


def sub(old, new, what):
    if t.count(old) != 1:
        raise SystemExit("anchor %s count %d" % (what, t.count(old)))
    return t.replace(old, new)


MMA_OLD = """                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }"""

MMA_NEW = """                        for (int mi = 0; mi < 4; ++mi) {
                            // ABLATION: the tensor instruction is replaced by a cheap dependency
                            // that still consumes all six operand registers, so ldmatrix and the
                            // staging feeding it survive dead-code elimination.
                            const float mix = __int_as_float(
                                (af[mi][0] ^ af[mi][1] ^ af[mi][2] ^ af[mi][3] ^ bf[ni][0] ^
                                 bf[ni][1]) &
                                0x3f800000u);
                            partial[mi][ni][0] += mix;
                            partial[mi][ni][1] += mix;
                            partial[mi][ni][2] += mix;
                            partial[mi][ni][3] += mix;
                        }"""

DEC_OLD = """                unsigned decoded[4];
                Q4MmaDecodeAtom::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]), decoded);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));"""

DEC_NEW = """                // ABLATION: the Q4 unpack arithmetic is replaced by a cheap function of the
                // staged codes. The shared load and the shared store both survive, so this
                // isolates the unpack itself from the traffic around it.
                const unsigned raw =
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]);
                const unsigned decoded0 = (raw & 0x03ff03ffu) | 0x3c003c00u;
                store_vec(&As[row * kExpertBK + gemm_swz64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded0), static_cast<int>(decoded0 ^ 1u),
                                    static_cast<int>(decoded0 ^ 2u),
                                    static_cast<int>(decoded0 ^ 3u)));"""

FOLD_OLD = """                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;
                    float scale0 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row0 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    float scale1 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row1 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    scale0 = __shfl_sync(0xffffffffu, scale0, gid * 4);
                    scale1 = __shfl_sync(0xffffffffu, scale1, gid * 4);
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }"""

FOLD_ABL = """                // ABLATION: the per-k-tile scale application is dropped. The accumulator
                // chain stays, so the register pressure and the fp32 adds remain comparable;
                // what disappears is the pair of shuffles and the Sr reads.
                for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] += partial[mi][ni][0];
                        acc[mi][ni][1] += partial[mi][ni][1];
                        acc[mi][ni][2] += partial[mi][ni][2];
                        acc[mi][ni][3] += partial[mi][ni][3];
                    }
                }"""

FOLD_SHFL = """                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;
                    // Every lane of a quad needs the same scale, and a shared-memory read of one
                    // address by four lanes is a broadcast. Reading it directly costs the same
                    // transaction as the lid == 0 read did and removes the pair of shuffles.
                    const float scale0 =
                        __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                            &Sr[(row0 * GroupsPerRow + kt) * 2])));
                    const float scale1 =
                        __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                            &Sr[(row1 * GroupsPerRow + kt) * 2])));
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }"""

if MODE == "mma":
    t = sub(MMA_OLD, MMA_NEW, "mma")
elif MODE == "dec":
    t = sub(DEC_OLD, DEC_NEW, "decode")
elif MODE == "fold":
    t = sub(FOLD_OLD, FOLD_ABL, "fold")
elif MODE == "shfl":
    t = sub(FOLD_OLD, FOLD_SHFL, "fold->shfl")
else:
    raise SystemExit("mode must be mma|dec|fold|shfl")

F.write_text(t, encoding="utf-8")
print("q4 gate/up patched:", MODE)
