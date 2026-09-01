#!/usr/bin/env python3
"""Decode attention: choose the split count by how much machine there is, not by a keys-per-split ladder.

Both geometries set SmallTMaximumSplits so that splits x kv_heads == 340 CTAs, i.e. the ladder's
ceiling is already "fill the machine". But between the ceiling and the bottom tier the policy asks
for a fixed number of keys per split (64 / 128 / 256 / 480 by window range), and in the 16k..40k
band that leaves the GPU most of the way idle: at window 16800 it asks for 35 splits, which is
140 CTAs on 170 SMs, and the kernel reads KV at 993 GB/s against an 1807 GB/s measured ceiling.

The replacement asks for as many splits as the machine has, subject to each split still owning at
least one Bc=64 key tile. Below window 5440 that is identical to today's bottom tier, so short
contexts do not move at all.

The same expression now serves the host bound and the device choice. They were different functions
before -- the host took a monotone envelope (max over tiers) while the device picked a single tier,
so just above each tier boundary the host launched roughly twice the blocks the device activated.

KEYS_PER_SPLIT_FLOOR is the knob this experiment sweeps.
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
FLOOR = int(sys.argv[2]) if len(sys.argv) > 2 else 64


def edit(rel, old, new):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit("anchor count %d in %s:\n%s" % (text.count(old), rel, old))
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", rel)


# ------------------------------------------------------------------ device side
edit(
    "src/ops/softmax_attention/dense/causal_cache/small_t.cuh",
    """template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::SmallTSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::SmallTSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::SmallTSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::SmallTSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::SmallTSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}""",
    """// Take as many splits as the machine has, as long as each one still owns a whole key tile.
// SmallTMaximumSplits is set per geometry so that splits * kv_heads is one 340-CTA fill.
template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    constexpr int kMinSplits      = 4 * Geometry::SmallTSplitScale;
    constexpr int kKeysPerSplitLo = %d / Geometry::SmallTSplitScale;
    int splits                    = div_up(window, kKeysPerSplitLo);
    splits                        = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}""" % FLOOR,
)

# ------------------------------------------------------------------ host side
edit(
    "src/ops/softmax_attention/dense/causal_cache/small_t.cu",
    """template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}""",
    """// Twin of causal_small_t_default_splits on the device. The two used to disagree: this one took
// a monotone envelope over the tier ladder while the device picked one tier, so just above a tier
// boundary the launch carried about twice the blocks the device would activate. One expression now
// serves both, and it is monotone in window, so it remains an exact upper bound.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits      = 4 * Geometry::SmallTSplitScale;
    constexpr std::int32_t kKeysPerSplitLo = %d / Geometry::SmallTSplitScale;
    std::int32_t splits                    = div_up(window, kKeysPerSplitLo);
    splits                                 = (splits > kMinSplits) ? splits : kMinSplits;
    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}""" % FLOOR,
)

print("done, floor =", FLOOR)
