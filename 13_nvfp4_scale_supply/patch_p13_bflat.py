#!/usr/bin/env python3
"""Weight-scale box widened on the FLAT nvfp4 route only.

On the flat route one weight-scale stream per stage becomes four 256-byte requests instead of
sixty-four 16-byte ones, and the route gets faster. On the fused SwiGLU route, which issues two
such streams a fixed large stride apart, the same descriptor makes the kernel 9% slower with
nine times the run-to-run spread, so the fused route keeps its 16-byte rows. Apply on top of the
activation-tile half.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])


def edit(rel, replacements):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit("anchor count %d in %s:\n%s" % (text.count(old), rel, old))
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("patched", rel)


edit(
    "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh",
    [
        (
            """template <class Geometry, int BlockM>
Nvfp4W4a4TmaDescriptors make_nvfp4_w4a4_tma_descriptors(""",
            """// One stage of weight scales is a contiguous 1024-byte run. On the flat route -- one such
// stream per stage -- describing it as four 256-byte rows instead of sixty-four 16-byte ones
// leaves the bytes, their order and the shared image untouched and makes the route faster.
// The fused SwiGLU route issues two of these streams a fixed large stride apart and gets slower
// from the same descriptor, so it keeps 16-byte rows; both numbers are in the package notes.
inline constexpr std::uint32_t kNvfp4WeightScaleRowBytes = 256;

template <class Geometry, int BlockM>
Nvfp4W4a4TmaDescriptors make_nvfp4_w4a4_tma_descriptors(""",
        ),
        (
            """    descriptors.b_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, 16,
        kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE, "encode weight scales TMA");""",
            """    static_assert(kWeightScaleBytes % kNvfp4WeightScaleRowBytes == 0,
                  "weight scale plane must be a whole number of 256-byte rows");
    descriptors.b_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        kNvfp4WeightScaleRowBytes, kWeightScaleBytes / kNvfp4WeightScaleRowBytes,
        kNvfp4WeightScaleRowBytes, kNvfp4WeightScaleRowBytes, 4, CU_TENSOR_MAP_SWIZZLE_NONE,
        "encode weight scales TMA");""",
        ),
        (
            """                const int b_scale_row = ((row_begin / 128) * Geometry::kScaleTilesPerRow +
                                         k_tile * Schedule::kK64PerStage) *
                                        32;
                nvfp4_tma_load_2d(tensors.b_scales[stage], &descriptors.b_scales, 0, b_scale_row,
                                  &shared.full[stage]);""",
            """                const int b_scale_bytes = ((row_begin / 128) * Geometry::kScaleTilesPerRow +
                                           k_tile * Schedule::kK64PerStage) *
                                          32 * 16;
                nvfp4_tma_load_2d(tensors.b_scales[stage], &descriptors.b_scales, 0,
                                  b_scale_bytes / static_cast<int>(kNvfp4WeightScaleRowBytes),
                                  &shared.full[stage]);""",
        ),
    ],
)

print("done: weight box widened on the flat route only")
