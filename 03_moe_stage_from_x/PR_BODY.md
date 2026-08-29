Closes #ISSUE. Scope as agreed there: ...

`sparse_moe_prefill_gather_kernel` materialises one 2048-wide bf16 row for **every (token, expert)
assignment**. At top-8 of 256 that is eight copies of every token: a 4096-token slice stores 32,768
rows of 4 KiB, **128 MiB**, and an 8192-token chunk stores that twice because the plan slices at
4096 - all of which `sparse_moe_prefill_q4_gate_up_kernel` then reads straight back through
`stage_inputs`.

Every one of those rows is a byte-exact copy of a column of the layer input `x`. The GEMM does not
need the copy - it needs to know which token each packed column came from. On the Q4 routed codec
the copy kernel becomes an index kernel that publishes only that map, and the staging loop follows
the map into `x`.

**Process.** Opened as #ISSUE first and sent after your reply. Targeted at `master`, one commit on
``1fc1cb76``; every number below was measured at `1fc1cb76`. It is unrelated to #112 and #113.

## What changes

Two files. `src/ops/sparse_moe/prefill/sparse_moe_prefill.h` gains one aliased `Tensor` and its
comment; `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu` gains the index kernel and
rewrites the Q4 staging loop.

The gather wrote a row per assignment and published the column from thread 0 of a block. The index
kernel writes four bytes per assignment from one thread:

```cpp
+    const int packed =
+        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
+    packed_index[assignment] = packed;
+    packed_token[packed]     = token;
```

and the staging loop resolves its rows once per route job rather than per k-tile:

```cpp
+        const __nv_bfloat16* src_row[StageIters];
+#pragma unroll
+        for (int i = 0; i < StageIters; ++i) {
+            const int col = (tid + i * ExpertThreads) / StageChunks;
+            const int row = packed_token[col < cols ? begin + column_base + col : begin];
+            src_row[i]    = x + static_cast<std::int64_t>(row) * kHidden;
+        }
```

A thread stages the same columns of all `kHidden / kExpertBK` tiles, so the map is read once for the
whole job and the k-loop walks pointers. That is not only fewer instructions: it keeps the map out
of shared memory, and the shared footprint of this kernel is what its `__launch_bounds__(..., 3)`
depends on. See **Resources** below - it does not move.

**The W8 routed codec keeps the gather**, which leaves one untouched codec in the operator benchmark
as a control. Question 3 in the Issue is about whether you want the other half; it is built and
measured, and the answer is at the end.

## Operator benchmark

`ninfer_sparse_moe_bench --codec all --cache cold --execution eager --repeat 50`, eager because
prefill is not captured into a CUDA graph. **Two independent passes**, so a row that moves can be
told from a row that is noisy.

```
./build/bench/ninfer_sparse_moe_bench --codec all --tokens <T> --cache cold \
    --execution eager --repeat 50
```

| codec | T | master us | branch us | pass 1 | pass 2 |
| --- | --: | --: | --: | --: | --: |
| q4-q5 | 1024 | 780.29 | 751.62 | **1.0381** | **1.0409** |
| q4-q5 | 2048 | 1062.94 | 1013.76 | **1.0485** | **1.0490** |
| q4-q5 | 4096 | 1995.78 | 1902.59 | **1.0490** | **1.0488** |
| q4-q5 | 8192 | 4017.86 | 3838.59 | **1.0467** | **1.0477** |
| q4-q6 | 1024 | 789.15 | 759.81 | **1.0386** | **1.0395** |
| q4-q6 | 2048 | 1069.06 | 1017.86 | **1.0503** | **1.0483** |
| q4-q6 | 4096 | 2010.11 | 1914.88 | **1.0497** | **1.0489** |
| q4-q6 | 8192 | 4041.89 | 3859.01 | **1.0474** | **1.0454** |
| w8-w8 | 1024 | 997.41 | 996.80 | 1.0006 | 0.9993 |
| w8-w8 | 2048 | 1376.26 | 1376.26 | 1.0000 | 1.0045 |
| w8-w8 | 4096 | 2491.17 | 2499.55 | 0.9966 | 0.9988 |
| w8-w8 | 8192 | 4978.37 | 4982.46 | 0.9992 | 0.9965 |

`w8-w8` is the same operator, the same benchmark and a codec this change does not reach. Its eight
cells span **0.9965 to 1.0045**, and that is the floor the Q4 rows should be read against.

The size of the gain is worth stating because it is also its ceiling. At T = 8192 the gather stores
256 MiB and reads about 32 MiB of `x` to do it. The measured saving is about 180 us of a 4018 us
operator, so those 288 MiB are leaving at roughly 1.6 TB/s - which is the traffic and nothing else.
**The GEMM itself does not get faster**; see the trace below.

## Workspace

`packed_token` needs no allocation. Scan is the final consumer of `tile_counts`, `packed_index`
already aliases its dead prefix, and the prefix is `256 * ceil(T / 8) >= 32 * T` elements against
`8 * T` per map, so both maps fit in half of it:

```cpp
     out.packed_index   = Tensor(out.tile_counts.data, DType::I32, {assignments});
+    out.packed_token   = Tensor(static_cast<std::int32_t*>(out.tile_counts.data) + assignments,
+                                DType::I32, {assignments});
```

No `arena.alloc` is added or removed, so the arena sequence is byte-identical to master's and
`sparse_moe_prefill_workspace_bytes` cannot move by construction. The benchmark reports it and
confirms it: `workspace_bytes` is **43,378,176 / 86,752,768 / 173,501,952 / 173,501,952** at
T = 1024 / 2048 / 4096 / 8192 on both arms, and the engine's own `gpu workspace peak` for the 35B
artifact at 131,072 context is 835.06 MiB on both arms with the same zero KV headroom.

`grouped_io` stays, and not because of the W8 gather. Your comment says why:

```cpp
    //   gathered X BF16    <-> routed down output BF16
```

The routed down projection writes its output into that buffer and the reduce reads it back, so it
does not leave even when nothing gathers into it. **This saves traffic, not capacity**, and I would
rather say that than quote a buffer that stays.

## Numerics

`gathered[packed]` was written as `x[token]` by the gather and `packed_token[packed]` resolves to
that same token, so the staged bytes are the same bytes in the same order. Columns past the tile
tail read the same fallback row the old code read and are zero-filled by the same `cp_async_zfill`
predicate. Same codes, same scales, same accumulation order.

The change also removes the read/write overlap that `3a61ef3f` separated the buffers for. That fix
exists because in the gather all threads of an assignment block consume the tile-local rank while
thread 0 publishes the packed column into the same place. The index kernel is one thread per
assignment: it reads the rank and writes the map itself, so the overlap does not arise on this path.
The fix stays where it is and keeps protecting the gather, which is still there for W8.

All 24 generations - four rounds over six points, two artifacts and two chunk widths - are
byte-identical to master, and so are the 24 from an earlier campaign on the first form of this
change.

## Resources

`cuobjdump --dump-resource-usage`, master against branch:

| instantiation | REG | SHARED | LOCAL | STACK |
| --- | --- | --- | --- | --- |
| `q4_gate_up<8, 64>` | 75 -> 80 | 33792 -> **33792** | 0 -> 0 | 0 -> 0 |
| `q4_gate_up<4, 32>` | 116 -> 124 | 25600 -> **25600** | 0 -> 0 | 0 -> 0 |
| `gather_kernel<0/1>` | 20, unchanged | 0 | 0 | 0 |
| `index_kernel<0/1>` | new, 16 | 0 | 0 | 0 |

The other 17 `sparse_moe_prefill` instantiations are untouched in all four columns.

Shared is the number that matters here, because `__launch_bounds__(ExpertWarps * 32, 3)` asks for
three blocks per SM. This device reports `sharedMemPerMultiprocessor` 102400 and
`reservedSharedMemPerBlock` 1024, so master's `<8, 64>` already sits at 33792 x 3 = 101376 with
1024 bytes of headroom. **The branch sits at exactly the same number**: the map lives in registers,
not in a shared tile. Registers are 80 x 256 x 3 = 61440 against 65536, and local, spill and stack
are zero on both sides.

I did try the shared tile first - it is the obvious way to write this - and it costs
`ExpertBN * sizeof(int)`, taking the headroom from 1024 bytes to 256. It was also slower in both
passes. Reading the map inline in the staging loop, which is the smallest diff of the three, was
slower still, by 0.4 to 1.4% on every Q4 point, because the map and the 64-bit row multiply are then
redone for each of the 32 k-tiles.

## Trace

`nsys profile -t cuda --cuda-graph-trace=node`, Qwen3.6-35B-A3B, 32K prompt, chunk 8192, one pass
per arm:

| master ms | branch ms | delta | kernel |
| --: | --: | --: | --- |
| 1948.18 | 1944.80 | -3.38 | `causal_attention_prompt_bf16_kernel` |
| 704.95 | 700.88 | **-4.07** | `sparse_moe_prefill_q4_gate_up_kernel<8, 64>` |
| 560.79 | 560.17 | -0.62 | `w8_rowsplit_gemm_mma_kernel`, widest |
| 359.05 | 353.12 | -5.93 | `sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 64>` |
| 95.20 | 92.52 | -2.68 | `sparse_moe_prefill_w8_gate_up_kernel<false, false>` |
| **57.61** | **0.00** | **-57.61** | `sparse_moe_prefill_gather_kernel<false>` |
| 4738.7 | 4663.0 | **-75.7** | total, all kernels |

The gather is 76% of what is saved. The GEMM that stopped reading it moves by 0.6%, which is inside
what a single pass resolves. The two other kernels that move - the routed down and the shared
gate/up - are the ones that no longer share the machine with 256 MiB of stores. Nothing else moves.
This is one pass per arm and is attribution, not measurement.

## Tests

```
cd build && ctest -j1
```

`100% tests passed, 0 tests failed out of 94`, 1 skipped, on this branch and on `master` built in
the same directory in the same run. The skip is `27b_load_plan`, which needs both real 27B artifacts
and only one is on this box.

## End to end

Confirmation only; the operator benchmark above is the claim. Arms alternate inside each round,
every point its own process, greedy, four rounds, on an otherwise idle box.

```
./build/apps/ninfer <artifact> --messages <prompt> --max-new 32 --greedy --no-thinking \
    --max-context 131072 --prefill-chunk <chunk> --kv-dtype bf16
```

| model | chunk | prompt | per-round branch/master | prefill | decode |
| --- | --: | --: | --- | --: | --: |
| Qwen3.6-35B-A3B | 1024 | 4,357 | 1.0215 1.0220 1.0224 1.0210 | **+2.17%** | +0.01% |
| Qwen3.6-35B-A3B | 1024 | 16,441 | 1.0180 1.0191 1.0181 1.0192 | **+1.86%** | -0.01% |
| Qwen3.6-35B-A3B | 1024 | 33,031 | 1.0141 1.0154 1.0144 1.0149 | **+1.47%** | -0.01% |
| Qwen3.6-35B-A3B | 8192 | 16,441 | 1.0195 1.0163 1.0207 1.0173 | **+1.85%** | -0.01% |
| Qwen3.6-35B-A3B | 8192 | 33,031 | 1.0151 1.0143 1.0149 1.0143 | **+1.47%** | +0.02% |
| Qwen3.6-27B dense | 8192 | 16,441 | 0.9950 1.0040 0.9954 1.0045 | -0.03% | -0.01% |

The dense 27B row is the null control and behaves like one: four ratios straddling 1.000, averaging
to -0.03%, on an artifact with no sparse MoE. I quote the paired per-round ratios rather than a mean
of means because the absolute rate drifts between rounds on both arms and the ratios do not.

All 24 generations are byte-identical to `master`.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
- The adaptive small-T route is reached by the same launcher branch and takes the index kernel too,
  but nothing in this campaign forces it; it is covered by `ctest` and not by a benchmark.
- Qwen3.8-27B NVFP4 takes no sparse-MoE prefill route and is not measured.
- Speculation is off in every run here.

## Question 3, answered by measurement

You asked - or I asked, and here is the answer - whether the W8 routed codec should get the same
treatment. I built it. The same mechanism in `sparse_moe_prefill_w8_gate_up_kernel`, guarded by its
existing `Routed` template parameter, gives `w8-w8`:

| T | 1024 | 2048 | 4096 | 8192 |
| --- | --: | --: | --: | --: |
| pass 1 | 1.0406 | 1.0673 | 1.0741 | 1.0742 |
| pass 2 | 1.0443 | 1.0707 | 1.0737 | 1.0711 |

That is a larger relative gain than Q4 sees, and it also makes the change smaller in one sense: with
both routed codecs staging from `x` the gather kernel has no caller left and the launcher stops
branching on the codec at all. The reason it is not in this pull request is that taking it removes
the only untouched codec in the benchmark above. Say the word and it is the same patch.

RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, Release, `-DCMAKE_CUDA_ARCHITECTURES=120a`.
