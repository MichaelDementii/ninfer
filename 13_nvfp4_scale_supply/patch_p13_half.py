#!/usr/bin/env python3
"""Package 13 split into its two independent halves, so each can be measured on its own.

  a = activation-scale tile shrinks from sixteen groups to eight (what a K128 stage consumes)
  b = weight-scale box changes from 16 B x 64 rows to 256 B x 4 rows over the same bytes
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MODE = sys.argv[2]
assert MODE in ("a", "b")


def edit(rel, replacements):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit("anchor count %d in %s:\n%s" % (text.count(old), rel, old))
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("patched", rel)


CONSTANT = (
    """template <class Geometry, int BlockM>
Nvfp4W4a4TmaDescriptors make_nvfp4_w4a4_tma_descriptors(""",
    """inline constexpr std::uint32_t kNvfp4WeightScaleRowBytes = 256;

template <class Geometry, int BlockM>
Nvfp4W4a4TmaDescriptors make_nvfp4_w4a4_tma_descriptors(""",
)

if MODE == "a":
    edit(
        "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh",
        [
            ("constexpr std::uint32_t kScaleTileGroups = 16;",
             "constexpr std::uint32_t kScaleTileGroups = 8;"),
            ("    static constexpr int kScaleWordsPerRow = 4;",
             "    static constexpr int kScaleWordsPerRow = 2;"),
            ("""                constexpr int kScaleTilesPerPlane = Geometry::kGroupsPerRow / 16;
                const int scale_tile =
                    (token_begin / Schedule::kBlockM) * kScaleTilesPerPlane + k_tile / 2;
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, 0,
                                  scale_tile * 16, &shared.full[stage]);""",
             """                constexpr int kScaleTilesPerPlane = Geometry::kGroupsPerRow / 8;
                const int scale_tile =
                    (token_begin / Schedule::kBlockM) * kScaleTilesPerPlane + k_tile;
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, 0, scale_tile * 8,
                                  &shared.full[stage]);"""),
            ("""                a_scales[mma_m] =
                    tensors.a_scale4[stage][scale_row * Schedule::kScaleWordsPerRow +
                                            (k_tile & 1) * Schedule::kK64PerStage + local_k64];""",
             """                a_scales[mma_m] =
                    tensors.a_scale4[stage][scale_row * Schedule::kScaleWordsPerRow + local_k64];"""),
        ],
    )
    edit(
        "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu",
        [("constexpr std::uint32_t kScaleTileGroups = 16;",
          "constexpr std::uint32_t kScaleTileGroups = 8;")],
    )
    edit(
        "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh",
        [
            ("""                constexpr int kScaleTilesPerPlane = Geometry::kGroupsPerRow / 16;
                const int scale_tile =
                    (token_begin / Schedule::kBlockM) * kScaleTilesPerPlane + k_tile / 2;
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, 0,
                                  scale_tile * 16, &shared.full[stage]);""",
             """                constexpr int kScaleTilesPerPlane = Geometry::kGroupsPerRow / 8;
                const int scale_tile =
                    (token_begin / Schedule::kBlockM) * kScaleTilesPerPlane + k_tile;
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, 0, scale_tile * 8,
                                  &shared.full[stage]);"""),
            ("""                a_scales[mma_m] =
                    tensors.a_scale4[stage][scale_row * Schedule::kScaleWordsPerRow +
                                            (k_tile & 1) * Schedule::kK64PerStage + local_k64];""",
             """                a_scales[mma_m] =
                    tensors.a_scale4[stage][scale_row * Schedule::kScaleWordsPerRow + local_k64];"""),
        ],
    )
    edit(
        "src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh",
        [("    constexpr int kGroupsPerTile = 16;", "    constexpr int kGroupsPerTile = 8;")],
    )
else:
    edit(
        "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh",
        [
            CONSTANT,
            ("""    descriptors.b_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, 16,
        kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE, "encode weight scales TMA");""",
             """    static_assert(kWeightScaleBytes % kNvfp4WeightScaleRowBytes == 0, "256-byte rows");
    descriptors.b_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        kNvfp4WeightScaleRowBytes, kWeightScaleBytes / kNvfp4WeightScaleRowBytes,
        kNvfp4WeightScaleRowBytes, kNvfp4WeightScaleRowBytes, 4, CU_TENSOR_MAP_SWIZZLE_NONE,
        "encode weight scales TMA");"""),
            ("""                const int b_scale_row = ((row_begin / 128) * Geometry::kScaleTilesPerRow +
                                         k_tile * Schedule::kK64PerStage) *
                                        32;
                nvfp4_tma_load_2d(tensors.b_scales[stage], &descriptors.b_scales, 0, b_scale_row,
                                  &shared.full[stage]);""",
             """                const int b_scale_bytes = ((row_begin / 128) * Geometry::kScaleTilesPerRow +
                                           k_tile * Schedule::kK64PerStage) *
                                          32 * 16;
                nvfp4_tma_load_2d(tensors.b_scales[stage], &descriptors.b_scales, 0,
                                  b_scale_bytes / static_cast<int>(kNvfp4WeightScaleRowBytes),
                                  &shared.full[stage]);"""),
        ],
    )
    edit(
        "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu",
        [("""    descriptors.b_scales =
        nvfp4_make_tma_2d(const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          16, kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                          "encode LinearSwiGLU weight scales TMA");""",
          """    static_assert(kWeightScaleBytes % kNvfp4WeightScaleRowBytes == 0, "256-byte rows");
    descriptors.b_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        kNvfp4WeightScaleRowBytes, kWeightScaleBytes / kNvfp4WeightScaleRowBytes,
        kNvfp4WeightScaleRowBytes, kNvfp4WeightScaleRowBytes, 4, CU_TENSOR_MAP_SWIZZLE_NONE,
        "encode LinearSwiGLU weight scales TMA");""")],
    )
    edit(
        "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh",
        [("""                const int gate_scale_row = ((pair_begin / 128) * Geometry::kScaleTilesPerRow +
                                            k_tile * Schedule::kK64PerStage) *
                                           32;
                const int up_scale_row =
                    (((pair_begin + kIntermediate) / 128) * Geometry::kScaleTilesPerRow +
                     k_tile * Schedule::kK64PerStage) *
                    32;""",
          """                constexpr int kScaleRowBytes = static_cast<int>(kNvfp4WeightScaleRowBytes);
                const int gate_scale_row = ((pair_begin / 128) * Geometry::kScaleTilesPerRow +
                                            k_tile * Schedule::kK64PerStage) *
                                           32 * 16 / kScaleRowBytes;
                const int up_scale_row =
                    (((pair_begin + kIntermediate) / 128) * Geometry::kScaleTilesPerRow +
                     k_tile * Schedule::kK64PerStage) *
                    32 * 16 / kScaleRowBytes;""")],
    )

print("done", MODE)
