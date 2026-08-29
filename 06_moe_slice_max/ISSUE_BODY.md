**Level of the claim: operator.** The evidence is `ninfer_sparse_moe_bench`; the end-to-end figures
are confirmation. **This changes a public workspace boundary** -
`sparse_moe_prefill_workspace_bytes(max_tokens)` returns twice as much for `max_tokens > 4096` - so
the cost is measured before the gain, and with two numbers rather than one.

**I am not proposing to change the default prefill chunk.** When you closed #96 you wrote that 1024
is deliberate and that the workspace growth at wider chunks comes out of KV capacity. The default
stays exactly where it is here, and not by promise: the constant is only ever read inside a `min()`,
so for any caller at or below 4096 both expressions return what they returned before. What changes
is only what a caller who has *already* chosen a chunk wider than 4096 gets.

## The observation

`resolve_sparse_moe_prefill_plan` slices the prefill chunk at a fixed 4096 tokens whatever the
caller asked for:

```cpp
inline constexpr std::int32_t kSparseMoePrefillSliceMax = 4096;
...
const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);  // workspace
const std::int32_t slice_tokens    = std::min(tokens,     kSparseMoePrefillSliceMax);  // execution
```

A caller that asks for an 8192-token chunk therefore gets two independent 4096-token passes through
the whole routed MoE. Each pass routes, scans and runs its own grouped GEMM, so **every expert's
column run is half as long as the chunk would allow** - and the length of that run is the economy of
the grouped route, because a route job re-reads its expert's entire weight slab and the average run
is `tokens / 32` at top-8 of 256.

Nothing in the kernels requires 4096. The constant bounds the workspace, not correctness.

## The change

```diff
-inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 4096;
+inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 8192;
```

## The operator benchmark is its own control

Because the constant only acts through `min()`, every width at or below 4096 has to read 1.000 with
an unchanged `workspace_bytes`, and only 8192 may move. It does:

```
./build/bench/ninfer_sparse_moe_bench --codec all --tokens <T> --cache cold \
    --execution eager --repeat 50
```

| codec | T | master us | branch us | speedup | master ws | branch ws |
| --- | --: | --: | --: | --: | --: | --: |
| q4-q5 | 1024 | 780.29 | 781.73 | 0.9982 | 43,378,176 | 43,378,176 |
| q4-q5 | 2048 | 1064.96 | 1064.96 | 1.0000 | 86,752,768 | 86,752,768 |
| q4-q5 | 4096 | 1997.41 | 1997.82 | 0.9998 | 173,501,952 | 173,501,952 |
| q4-q5 | 8192 | 4019.84 | 3765.25 | **1.0676** | 173,501,952 | **347,000,320** |
| q4-q6 | 1024 | 789.28 | 788.48 | 1.0010 | 43,378,176 | 43,378,176 |
| q4-q6 | 2048 | 1069.06 | 1069.06 | 1.0000 | 86,752,768 | 86,752,768 |
| q4-q6 | 4096 | 2010.02 | 2012.16 | 0.9989 | 173,501,952 | 173,501,952 |
| q4-q6 | 8192 | 4047.90 | 3783.36 | **1.0699** | 173,501,952 | **347,000,320** |
| w8-w8 | 1024 | 997.38 | 997.38 | 1.0000 | 43,378,176 | 43,378,176 |
| w8-w8 | 2048 | 1378.30 | 1380.35 | 0.9985 | 86,752,768 | 86,752,768 |
| w8-w8 | 4096 | 2493.47 | 2493.34 | 1.0001 | 173,501,952 | 173,501,952 |
| w8-w8 | 8192 | 4980.42 | 4595.65 | **1.0837** | 173,501,952 | **347,000,320** |

Nine of the twelve rows are the control and they span 0.9982 to 1.0010 with `workspace_bytes` equal
to the byte. All three codecs gain at 8192, and the widest gain is on `w8-w8`, which is the codec
none of my other submissions touch.

## What it costs, in two numbers

**One: the operator's own query doubles at 8192**, 165.5 MiB to 330.9 MiB, exactly as the table
above reports. That is a public boundary and it is the reason this is an Issue rather than a small
patch.

**Two: the engine's arena peak does not move.** Qwen3.6-35B-A3B, `--max-context 131072`, both arms:

| prefill chunk | master `gpu workspace peak` | branch |
| --: | --: | --: |
| 1024 | 104.38 MiB | 104.38 MiB |
| 4096 | 417.53 MiB | 417.53 MiB |
| 8192 | 835.06 MiB | 835.06 MiB |

Identical at every width, and the reason is checkable rather than lucky: at chunk 8192 the arena
holds 835 MiB and the MoE prefill asks 331 MiB of it even after doubling, so the MoE is not what
sets the peak at that width. The peak scales with the chunk on `master` already - 104.38 x 4 =
417.5, x 8 = 835.0 - and that scaling is everything else, not this constant.

**The limitation, before anyone has to ask:** that second number is one artifact. On a model where
the sparse-MoE prefill workspace *is* the arena's largest consumer at the configured chunk, the peak
would move by exactly the delta the query reports. I measured one model and I am not generalising
past it.

For completeness against the figures in #96: you quoted about 120 MiB at 1024, 482 at 4096 and 963
at 8192. On current `master` this box reports 104.38 / 417.53 / 835.06 for the same three chunks,
and the branch reports the same three. Whatever produced 963 at 8192 in that stack, it was not this
constant on its own.

## Values are unchanged

Slice width divides the token range and changes no reduction: the grouped GEMM accumulates over the
hidden dimension, and a wider slice adds columns rather than reordering a sum. Greedy output is
byte-identical - 16 generations over four rounds, four points, two chunk widths.

A host-side constant should also move no device code, and `cuobjdump -sass` says it moves none: of
2937 device functions, 2899 are comparable and **0 differ**, including all 19 `sparse_moe_prefill`
bodies. The 38 that cannot be compared are internal-linkage instantiations whose symbols carry a
per-compilation module id.

## Measured effect

`ctest -j1` is `100% tests passed, 0 tests failed out of 94`, 1 skipped, on the branch.

**Which bucket moves**, `nsys profile -t cuda --cuda-graph-trace=node`, 35B-A3B, 32K prompt, chunk
8192, one pass per arm:

| bucket | master ms | branch ms | delta |
| --- | --: | --: | --: |
| attention | 1945.6 | 1946.0 | +0.5 |
| **MoE** | **1449.1** | **1382.0** | **-67.0** |
| projections | 987.2 | 986.5 | -0.7 |
| GDN | 273.4 | 273.5 | +0.1 |
| other | 72.1 | 70.6 | -1.5 |
| total | 4727.3 | 4658.7 | -68.6 |

The MoE bucket is -67.0 of the -68.6; everything else is inside half a percent, which is what a
change that only lengthens expert column runs should look like. Inside the bucket, three kernels pay
for two:

| kernel | master ms | branch ms | delta |
| --- | --: | --: | --: |
| `q4_gate_up<8, 64>` | 702.14 | 655.81 | **-46.33** |
| `qx_down<Q5DownMma, 8, 64>` | 357.69 | 340.34 | **-17.35** |
| `w8_gate_up<false, false>` | 95.01 | 86.49 | **-8.52** |
| `gather_kernel<false>` | 57.60 | 64.03 | +6.43 |
| `reduce_kernel<false>` | 56.27 | 62.11 | +5.84 |

The gather and the reduce get **slower**: one launch over 8192 tokens against two over 4096 each is
worse for them, and I would rather show that than net it away.

**End to end**, confirmation only. Arms alternate inside each round, every point its own process,
greedy, four rounds:

| chunk | prompt | per-round branch/master | prefill | decode |
| --: | --: | --- | --: | --: |
| 1024 | 33,031 | 0.9990 1.0009 0.9984 1.0022 | +0.01% | -0.02% |
| 4096 | 33,031 | 0.9987 1.0008 0.9994 1.0022 | +0.03% | +0.00% |
| 8192 | 16,441 | 1.0209 1.0202 1.0189 1.0214 | **+2.04%** | +0.01% |
| 8192 | 33,031 | 1.0151 1.0176 1.0148 1.0170 | **+1.61%** | -0.01% |

The first two rows are controls by construction, not by luck, and they read +0.01% and +0.03%.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
- **The deepest end-to-end point on this box is 33K.** The 64K fixture fails identically on both
  arms with `prepared prompt exceeds Engine max_context 131072` - it prepares longer than its name
  suggests - and `--max-context` cannot go higher because KV headroom at 131072 is already zero.
- One `nsys` pass per arm; the bucket table is attribution, not measurement.
- The arena-peak result is one artifact, as stated above.
- `docs/performance.md` publishes prefill throughput at a 1024-token chunk, which this change leaves
  untouched by construction; I have not re-run that harness and propose no edits to the document.
- Speculation is off, KV is bf16 in every end-to-end run here.

## Questions

1. Is 8192 the right ceiling, or should it stop being a constant? It bounds the workspace, not the
   kernels, so the honest form might be to derive it from the largest prefill chunk the engine is
   configured for. I did not do that because a 32768-token chunk would then ask for about 1.3 GiB of
   MoE workspace, and where that line goes is your call.
2. Is the arena-peak observation something you want in the submission at all, or would you rather
   the report stopped at the query? It is the more useful number for a reader and the less
   conservative one, which is exactly why I am asking rather than deciding.
