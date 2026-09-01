#!/usr/bin/env python3
"""Decode attention: hand every split the same whole number of key tiles, and use the whole machine.

Today the split count comes from a keys-per-split ladder (64 / 128 / 256 / 480 by window range).
Two things go wrong with it.

  * Between the ladder's bottom tier and its ceiling the machine sits idle. At window 16800 the
    ladder asks for 35 splits, which is 140 CTAs on 170 SMs, and the kernel reads KV at 993 GB/s
    against the 1807 GB/s the same kernel reaches when the split count fills the grid.
  * The host and the device disagreed. The host took a monotone envelope (max over tiers) while
    the device picked a single tier, so just above a tier boundary the launch carried about twice
    the blocks the device would activate.

SmallTMaximumSplits is already set per geometry so that splits * kv_heads is one 340-CTA fill, so
"fill the machine" is the ladder's own ceiling. What is missing is the shape in between: give each
split ceil(tiles / max_splits) tiles, then take the fewest splits that still cover every tile.
That never exceeds the ceiling, never leaves a split holding a partial tile, and is monotone in
window, so one expression serves both the host bound and the device choice.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])


def edit(rel, old, new):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit("anchor count %d in %s:\n%s" % (text.count(old), rel, old))
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", rel)


edit(
    "src/ops/softmax_attention/dense/causal_cache/small_t.cuh",
    """// Take as many splits as the machine has, as long as each one still owns a whole key tile.
// SmallTMaximumSplits is set per geometry so that splits * kv_heads is one 340-CTA fill.
template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    constexpr int kMinSplits      = 4 * Geometry::SmallTSplitScale;
    constexpr int kKeysPerSplitLo = 64 / Geometry::SmallTSplitScale;
    int splits                    = div_up(window, kKeysPerSplitLo);
    splits                        = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}""",
    """// Every split gets the same whole number of key tiles, and the count fills the machine:
// SmallTMaximumSplits is set per geometry so that splits * kv_heads is one 340-CTA fill.
template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    constexpr int kMinSplits    = 4 * Geometry::SmallTSplitScale;
    constexpr int kKeysPerTile  = 64 / Geometry::SmallTSplitScale;
    constexpr int kMaximumSplits = Geometry::SmallTMaximumSplits;
    const int tiles             = div_up(window, kKeysPerTile);
    const int tiles_per_split   = div_up(tiles, kMaximumSplits);
    int splits                  = div_up(tiles, tiles_per_split);
    splits                      = splits > kMinSplits ? splits : kMinSplits;
    return splits < kMaximumSplits ? splits : kMaximumSplits;
}""",
)

edit(
    "src/ops/softmax_attention/dense/causal_cache/small_t.cu",
    """// Twin of causal_small_t_default_splits on the device. The two used to disagree: this one took
// a monotone envelope over the tier ladder while the device picked one tier, so just above a tier
// boundary the launch carried about twice the blocks the device would activate. One expression now
// serves both, and it is monotone in window, so it remains an exact upper bound.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits      = 4 * Geometry::SmallTSplitScale;
    constexpr std::int32_t kKeysPerSplitLo = 64 / Geometry::SmallTSplitScale;
    std::int32_t splits                    = div_up(window, kKeysPerSplitLo);
    splits                                 = (splits > kMinSplits) ? splits : kMinSplits;
    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}""",
    """// Twin of causal_small_t_default_splits on the device. The two used to disagree: this one took
// a monotone envelope over the tier ladder while the device picked one tier, so just above a tier
// boundary the launch carried about twice the blocks the device would activate. One expression now
// serves both, and it is monotone in window, so it remains an exact upper bound.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits     = 4 * Geometry::SmallTSplitScale;
    constexpr std::int32_t kKeysPerTile   = 64 / Geometry::SmallTSplitScale;
    constexpr std::int32_t kMaximumSplits = Geometry::SmallTMaximumSplits;
    const std::int32_t tiles              = div_up(window, kKeysPerTile);
    const std::int32_t tiles_per_split    = div_up(tiles, kMaximumSplits);
    std::int32_t splits                   = div_up(tiles, tiles_per_split);
    splits                                = (splits > kMinSplits) ? splits : kMinSplits;
    return (splits < kMaximumSplits) ? splits : kMaximumSplits;
}""",
)

print("done: balanced whole tiles")
