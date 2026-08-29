**Level of the claim: operator.** The evidence is `ninfer_sparse_moe_bench`; the end-to-end figures
are confirmation. No public contract moves: the workspace query, the arena sequence and every op
signature are untouched. What moves is occupancy, deliberately, and that is measured below.

## The observation

The grouped routed GEMMs run one job per expert column tile, and the tile is 64 columns wide at
every chunk width:

```cpp
        const bool wide_plan   = tokens >= kSparseMoePrefillWideMin;
        const int route_job_bn = wide_plan ? 64 : 32;
```

Each job re-reads its expert's entire weight slab. At top-8 of 256 the average column run per expert
is `tokens / 32`, so a 64-wide job covers a 2048-token slice exactly, re-reads every slab twice at
4096 and four times at 8192. The weight traffic of the routed MoE is therefore a function of the job
width and the slice width together, and only one of the two follows the caller.

## The change

The kernels are already templated on `ExpertBN`; what is missing is a 128-wide instantiation and a
rule for choosing it. Both routed GEMMs get one, and the rule is the chunk:

```cpp
+inline constexpr std::int32_t kSparseMoePrefillWideBnTokens = 4096;
...
-        const int route_job_bn = wide_plan ? 64 : 32;
+        const bool wide_bn     = tokens >= kSparseMoePrefillWideBnTokens;
+        const int route_job_bn = wide_plan ? (wide_bn ? 128 : 64) : 32;
```

`sparse_moe_prefill_qx_down_kernel` needed generalising to reach 128 - its accumulator, its B
fragments and its epilogue were written for exactly one 8-column group per warp, the way
`sparse_moe_prefill_q4_gate_up_kernel` already is not.

## Occupancy is the price, and it is arithmetic rather than a choice

A 128-wide tile doubles `Bs`, and three blocks per SM stop fitting. One expression owns both the
launch bound and the grid so they cannot drift apart:

```cpp
constexpr int prefill_blocks_per_sm(int expert_bn) { return expert_bn >= 128 ? 2 : 3; }
constexpr int prefill_persistent_blocks(int expert_bn) {
    return prefill_blocks_per_sm(expert_bn) * kRtx5090SmCount;
}
```

The numbers under it, from `cuobjdump --dump-resource-usage` on a `master` binary against a device
that reports `sharedMemPerMultiprocessor` 102400 and `reservedSharedMemPerBlock` 1024:

| kernel | shared at BN=64 | at BN=128 | x2 blocks | x3 blocks |
| --- | --: | --: | --: | --: |
| `q4_gate_up<8, ·>` | 33792 | 50176 | 100352 | 150528, does not fit |
| `qx_down<Q5DownMma, 8, ·>` | 31744 | 48128 | 96256 | 144384, does not fit |
| `qx_down<Q6DownMma, 8, ·>` | 32768 | 49152 | 98304 | 147456, does not fit |

TBD_RESOURCES

## Measured effect

TBD_MEASURED

## Values are unchanged

Job width divides the column range and changes no reduction: each job accumulates over the hidden
dimension for its own columns, and a wider job carries more columns rather than reordering a sum.
TBD_GATE

## Checks not run

TBD_LIMITS

## Questions

1. **Two default numbers, and both need your opinion rather than mine.** 128 as the wide tile, and
   4096 as the width above which it is chosen. I can justify 4096 - it is where a 64-wide job starts
   re-reading each slab more than twice - but the sweep around it is below and the shape of that
   curve is the real argument.
2. Occupancy drops from three blocks per SM to two on the routed GEMMs whenever the wide tile is
   selected. That is forced by shared memory, not chosen, but it is still a schedule change on a
   path you own; is that acceptable, or would you rather the wide tile came with a narrower
   `ExpertWarps` so three blocks still fit?
3. This is written from scratch rather than cherry-picked. Two earlier commits of ours did the same
   thing through an `NINFER_MOE_BN` environment variable and two launch macros; neither is here,
   because a hidden mutable channel in production code is exactly what you told us not to bundle.
