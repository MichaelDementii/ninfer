#!/usr/bin/env python3
"""Probe: what is the f16 accumulator worth on the PV product of the bf16 prefill attention.

The correct version has to fold the f16 buffer into f32 once per key block and swap the PV loop
so that only the current n-tile's buffer is live. That is a restructuring. This probe keeps the
loop exactly as it is and reuses one shared two-register buffer, so the instruction count, the
operands and the ldmatrix pipeline are unchanged and only the tensor instruction differs.

Results are wrong on purpose. The number this produces is the ceiling of the direction.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])

MMA = ROOT / "src/ops/common/mma.cuh"
text = MMA.read_text(encoding="utf-8")

helper = (
    "// PROBE ONLY: f16 operands with an f16 accumulator. Measured tier 506.6 against 255.3 for\n"
    "// the f32-accumulate form. Correctness needs a fold into f32 once per key block; this exists\n"
    "// to measure what the tier is worth before that restructuring is written.\n"
    "__device__ __forceinline__ void mma_f16_acc16(unsigned& d0, unsigned& d1, unsigned a0,\n"
    "                                              unsigned a1, unsigned a2, unsigned a3,\n"
    "                                              unsigned b0, unsigned b1) {\n"
    '    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "\n'
    '                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"\n'
    '                 : "+r"(d0), "+r"(d1)\n'
    '                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n'
    "}\n\n"
)

anchor = "__device__ __forceinline__ void mma_fp8_e4m3("
if text.count(anchor) != 1:
    raise SystemExit("mma_fp8_e4m3 anchor count %d" % text.count(anchor))
MMA.write_text(text.replace(anchor, helper + anchor), encoding="utf-8")
print("helper added")

KERNEL = ROOT / "src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
src = KERNEL.read_text(encoding="utf-8")

old_mma = """            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);"""

new_mma = """            // PROBE ONLY: identical instruction count and operands, f16 accumulator, one shared
            // buffer so the register file barely moves. Results are wrong on purpose.
            mma_f16_acc16(probe_acc16[0], probe_acc16[1], p_frag[k][0], p_frag[k][1], p_frag[k][2],
                          p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_f16_acc16(probe_acc16[0], probe_acc16[1], p_frag[k][0], p_frag[k][1], p_frag[k][2],
                          p_frag[k][3], vf[cur][2], vf[cur][3]);
            acc[n2][0] += __half2float(__ushort_as_half(static_cast<unsigned short>(probe_acc16[0])));
            acc[n2 + 1][0] +=
                __half2float(__ushort_as_half(static_cast<unsigned short>(probe_acc16[1])));"""

if src.count(old_mma) != 1:
    raise SystemExit("PV mma anchor count %d" % src.count(old_mma))
src = src.replace(old_mma, new_mma)

decl_anchor = "        constexpr int PVHalf  = PVNt / 2;      // 16 n-tile pairs"
if src.count(decl_anchor) != 1:
    raise SystemExit("PVHalf anchor count %d" % src.count(decl_anchor))
src = src.replace(decl_anchor, "        unsigned probe_acc16[2] = {0u, 0u};\n" + decl_anchor)

KERNEL.write_text(src, encoding="utf-8")
print("probe wired into PV loop")
