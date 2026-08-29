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

And the three instantiations the change adds, from `cuobjdump --dump-resource-usage` on the branch
binary — the numbers land on the table above to the byte:

| new instantiation | REG | SHARED | LOCAL | STACK |
| --- | --: | --: | --- | --- |
| `q4_gate_up<8, 128>` | 126 | 50176 | 0 | 0 |
| `qx_down<Q5DownMma, 8, 128>` | 114 | 48128 | 0 | 0 |
| `qx_down<Q6DownMma, 8, 128>` | 116 | 49152 | 0 | 0 |

Registers hold two blocks as well, and only just: 126 x 256 x 2 = 64512 against 65536, two to spare
on the gate/up. Nothing spills.

**The 19 existing `sparse_moe_prefill` instantiations are unchanged in all four columns.** That
matters more than the new rows: `prefill_blocks_per_sm(64)` is 3, the value the literal used to be,
and the `WarpNT` generalisation collapses to the old shape at `ExpertBN == 64`, where `WarpNT` is 1.

## Measured effect

`ninfer_sparse_moe_bench --codec all --cache cold --execution eager --repeat 50`, eager because
prefill is not captured into a CUDA graph. **Two independent passes.**

| codec | T | master us | pass 1 | pass 2 |
| --- | --: | --: | --: | --: |
| q4-q5 | 1024 | 780.29 | 0.9994 | 0.9974 |
| q4-q5 | 2048 | 1062.94 | 0.9981 | 1.0003 |
| q4-q5 | 4096 | 1995.78 | **1.0592** | **1.0594** |
| q4-q5 | 8192 | 4017.41 | **1.0653** | **1.0673** |
| q4-q6 | 1024 | 788.51 | 1.0000 | 1.0000 |
| q4-q6 | 2048 | 1067.04 | 0.9984 | 1.0000 |
| q4-q6 | 4096 | 2008.06 | **1.0611** | **1.0637** |
| q4-q6 | 8192 | 4042.43 | **1.0677** | **1.0689** |
| w8-w8 | 1024 | 997.38 | 1.0006 | 0.9997 |
| w8-w8 | 2048 | 1378.30 | 1.0015 | 0.9981 |
| w8-w8 | 4096 | 2491.39 | 0.9992 | 1.0000 |
| w8-w8 | 8192 | 4977.98 | 1.0000 | 1.0003 |

Two controls, and both are controls by construction rather than by luck. The 1024 and 2048 rows sit
below `kSparseMoePrefillWideBnTokens`, so the plan cannot reach the wide tile there. And **`w8-w8`
never can**: the W8 routed codec goes through `sparse_moe_prefill_w8_gate_up_kernel` and
`sparse_moe_prefill_w8_down_kernel`, which are not the templates this change widens. Eight cells,
0.9981 to 1.0015, and that is the floor the Q4 rows should be read against.

`workspace_bytes` is identical to master at every width and every codec: 43,378,176 / 86,752,768 /
173,501,952 / 173,501,952. Nothing about the workspace moves.

`ctest -j1`: `100% tests passed, 0 tests failed out of 94`, 1 skipped.

**Which bucket moves**, `nsys profile -t cuda --cuda-graph-trace=node`, Qwen3.6-35B-A3B, 32K prompt,
chunk 8192, one pass per arm:

| bucket | master ms | branch ms | delta |
| --- | --: | --: | --: |
| attention | 1955.7 | 1954.3 | -1.4 |
| **MoE** | **1461.3** | **1379.2** | **-82.1** |
| projections | 992.5 | 988.9 | -3.6 |
| GDN | 274.5 | 274.3 | -0.2 |
| other | 72.3 | 72.1 | -0.2 |
| total | 4756.3 | 4668.8 | -87.5 |

Inside the bucket the two widened GEMMs are the whole of it, and the shared expert follows because
it stops sharing the machine with as many narrow jobs:

| kernel | master ms | branch ms |
| --- | --: | --: |
| `q4_gate_up<8, 64>` -> `<8, 128>` | 709.31 | **667.33** |
| `qx_down<Q5DownMma, 8, 64>` -> `<8, 128>` | 361.19 | **330.64** |
| `w8_gate_up<false, false>`, unchanged code | 95.48 | 92.12 |

**End to end**, confirmation only. Arms alternate inside each round, every point its own process,
greedy, four rounds:

| chunk | prompt | per-round branch/master | prefill | decode |
| --: | --: | --- | --: | --: |
| 1024 | 33,031 | 0.9986 1.0003 1.0003 1.0001 | -0.02% | +0.01% |
| 8192 | 16,441 | 1.0193 1.0209 1.0228 1.0215 | **+2.11%** | -0.00% |
| 8192 | 33,031 | 1.0152 1.0182 1.0177 1.0180 | **+1.73%** | -0.02% |

The chunk-1024 row is the control the predicate guarantees, and it reads -0.02% over four rounds.
The engine's `gpu workspace peak` is 104.38 MiB at chunk 1024 and 835.06 MiB at chunk 8192 on both
arms, unchanged.

## Values are unchanged

Job width divides the column range and changes no reduction: each job accumulates over the hidden
dimension for its own columns, and a wider job carries more columns rather than reordering a sum.
Greedy output is byte-identical: 12 generations over four rounds, three points, two chunk widths.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
- **I did not sweep the switch point.** `kSparseMoePrefillWideBnTokens` is 4096 because that is
  where a 64-wide job starts re-reading each expert slab more than twice, and the benchmark shows
  the wide tile winning at 4096 and 8192 and not being reachable below. What I have not done is
  measure 2048 and 3072 with the threshold lowered, which is the evidence that would actually
  justify the number rather than explain it. Question 1 below.
- No SASS census. The resource table says the 19 existing instantiations keep their registers,
  shared, local and stack, and the greedy gate says the output is unchanged, but I have not compared
  the instruction streams body by body.
- One `nsys` pass per arm; the bucket table is attribution, not measurement.
- `docs/performance.md` publishes prefill throughput at a 1024-token chunk, which this change cannot
  reach - the plan never selects the wide tile there. I have not re-run that harness.
- Qwen3.8-27B NVFP4 takes no sparse-MoE prefill route and is not measured. Speculation is off.

## Questions

1. **Two default numbers, and neither is measured into place.** 128 as the wide tile, and 4096 as
   the width above which it is chosen. I can explain 4096 - it is where a 64-wide job starts
   re-reading each expert slab more than twice - but explaining is not measuring, and I did not
   sweep the threshold. If you want that curve before deciding, say so and I will lower the constant
   and measure 2048 and 3072 on both tile widths.
2. Occupancy drops from three blocks per SM to two on the routed GEMMs whenever the wide tile is
   selected. That is forced by shared memory, not chosen, but it is still a schedule change on a
   path you own; is that acceptable, or would you rather the wide tile came with a narrower
   `ExpertWarps` so three blocks still fit?
3. **It composes with the slice ceiling more than multiplicatively**, which surprised me and may
   matter to how you want to see the two. Measured separately at chunk 8192, the slice ceiling is
   +2.04% and +1.61% end to end at 16K and 33K, and this change is +2.11% and +1.73%; together they
   read **+5.69% and +4.53%**, against a product of +4.19% and +3.37%. They are complementary rather
   than independent: the ceiling makes the column runs twice as long, and the wide tile is what has
   enough columns to consume them. The two edits also sit on adjacent lines of the same header, so
   whichever of them goes second needs a one-line rebase.
4. This is written from scratch rather than cherry-picked. Two earlier commits of ours did the same
   thing through an `NINFER_MOE_BN` environment variable and two launch macros; neither is here,
   because a hidden mutable channel in production code is exactly what you told us not to bundle.
