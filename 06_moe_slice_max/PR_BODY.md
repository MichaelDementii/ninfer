Closes #ISSUE. Scope as agreed there: ...

`resolve_sparse_moe_prefill_plan` slices the prefill chunk at a fixed 4096 tokens whatever the
caller asked for, so a caller that asks for an 8192-token chunk gets two independent passes through
the whole routed MoE and every expert's column run is half as long as the chunk would allow. That
run length is the economy of the grouped route: a route job re-reads its expert's entire weight
slab, and the average run is `tokens / 32` at top-8 of 256.

**Process.** Opened as #ISSUE first and sent after your reply. Targeted at `master`, one commit on
`TBD_BASE`; every number here was measured at `1fc1cb76`. **The default prefill chunk is untouched**
- see the next section, it is a property of the expression rather than a promise. It is unrelated to
#112 and #113.

## What changes

One line, `src/ops/sparse_moe/prefill/sparse_moe_prefill.h`:

```diff
-inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 4096;
+inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 8192;
```

The constant is read in exactly two places, both inside a `min()`:

```cpp
const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);  // workspace
const std::int32_t slice_tokens    = std::min(tokens,     kSparseMoePrefillSliceMax);  // execution
```

so every caller at or below 4096 tokens gets the same plan, the same workspace and the same
arithmetic as before. The default 1024-token chunk cannot move.

## Operator benchmark

That property makes the benchmark its own control: the rows at 1024, 2048 and 4096 must read 1.000
with an unchanged `workspace_bytes`, and only 8192 may move.

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

Nine of the twelve rows are the control and span 0.9982 to 1.0010, with `workspace_bytes` equal to
the byte. All three codecs gain at 8192.

## Workspace

**The operator's own query doubles at 8192**, 165.5 MiB to 330.9 MiB, which the table above reports
directly. `sparse_moe_prefill_workspace_bytes(max_tokens)` is public, so this is a contract change
for `max_tokens > 4096` and for nothing else.

**The engine's arena peak does not move.** Qwen3.6-35B-A3B, `--max-context 131072`, both arms:

| prefill chunk | master `gpu workspace peak` | branch |
| --: | --: | --: |
| 1024 | 104.38 MiB | 104.38 MiB |
| 4096 | 417.53 MiB | 417.53 MiB |
| 8192 | 835.06 MiB | 835.06 MiB |

The reason is checkable rather than lucky: at chunk 8192 the arena holds 835 MiB and the MoE prefill
asks 331 MiB of it even after doubling, so the MoE is not what sets the peak at that width. The peak
already scales with the chunk on `master` - 104.38 x 4 = 417.5, x 8 = 835.0 - and that scaling is
everything else, not this constant.

**Limitation:** this second number is one artifact. On a model where the sparse-MoE prefill
workspace *is* the arena's largest consumer at the configured chunk, the peak would move by exactly
the delta the query reports.

## Numerics

Slice width divides the token range and changes no reduction: the grouped GEMM accumulates over the
hidden dimension, and a wider slice adds columns rather than reordering a sum.

A host-side constant should move no device code, and it moves none:

```
cuobjdump -sass, master binary against branch binary
functions 2937 / 2937, comparable 2899, bodies differing: 0
sparse_moe_prefill bodies: 19, of which changed: 0
```

Greedy output is byte-identical: 16 generations over four rounds, four points, two chunk widths.

## Tests

```
cd build && ctest -j1
```

`100% tests passed, 0 tests failed out of 94`, 1 skipped. The skip is `27b_load_plan`, which needs
both real 27B artifacts and only one is on this box.

## Which bucket moves

`nsys profile -t cuda --cuda-graph-trace=node`, 35B-A3B, 32K prompt, chunk 8192, one pass per arm:

| bucket | master ms | branch ms | delta |
| --- | --: | --: | --: |
| attention | 1945.6 | 1946.0 | +0.5 |
| **MoE** | **1449.1** | **1382.0** | **-67.0** |
| projections | 987.2 | 986.5 | -0.7 |
| GDN | 273.4 | 273.5 | +0.1 |
| other | 72.1 | 70.6 | -1.5 |
| total | 4727.3 | 4658.7 | -68.6 |

Everything outside the MoE bucket is inside half a percent, which is what a change that only
lengthens expert column runs should look like. Inside the bucket, three kernels pay for two:

| kernel | master ms | branch ms | delta |
| --- | --: | --: | --: |
| `q4_gate_up<8, 64>` | 702.14 | 655.81 | **-46.33** |
| `qx_down<Q5DownMma, 8, 64>` | 357.69 | 340.34 | **-17.35** |
| `w8_gate_up<false, false>` | 95.01 | 86.49 | **-8.52** |
| `gather_kernel<false>` | 57.60 | 64.03 | +6.43 |
| `reduce_kernel<false>` | 56.27 | 62.11 | +5.84 |

The gather and the reduce get **slower** - one launch over 8192 tokens against two over 4096 each is
worse for them - and I would rather show that than net it away.

## End to end

Confirmation only. Arms alternate inside each round, every point its own process, greedy, four
rounds, on an otherwise idle box.

```
./build/apps/ninfer qwen3_6_35b_a3b.ninfer --messages <prompt> --max-new 32 --greedy \
    --no-thinking --max-context 131072 --prefill-chunk <chunk> --kv-dtype bf16
```

| chunk | prompt | per-round branch/master | prefill | decode |
| --: | --: | --- | --: | --: |
| 1024 | 33,031 | 0.9990 1.0009 0.9984 1.0022 | +0.01% | -0.02% |
| 4096 | 33,031 | 0.9987 1.0008 0.9994 1.0022 | +0.03% | +0.00% |
| 8192 | 16,441 | 1.0209 1.0202 1.0189 1.0214 | **+2.04%** | +0.01% |
| 8192 | 33,031 | 1.0151 1.0176 1.0148 1.0170 | **+1.61%** | -0.01% |

The first two rows are controls by construction rather than by luck, and they read +0.01% and
+0.03%. I quote the paired per-round ratios rather than a mean of means because the absolute rate
drifts between rounds on both arms and the ratios do not.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
- **The deepest end-to-end point on this box is 33K.** The 64K fixture fails identically on both
  arms with `prepared prompt exceeds Engine max_context 131072` - it prepares longer than its name
  suggests - and `--max-context` cannot go higher because KV headroom at 131072 is already zero.
- One `nsys` pass per arm; the bucket table is attribution, not measurement.
- The arena-peak result is one artifact, as stated above.
- `docs/performance.md` publishes prefill throughput at a 1024-token chunk, which this change leaves
  untouched by construction; I have not re-run that harness and propose no edits to the document.
- Speculation is off and KV is bf16 in every end-to-end run here.

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, Release, `-DCMAKE_CUDA_ARCHITECTURES=120a`.
