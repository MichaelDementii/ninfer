**Level of the claim: operator.** The evidence is `ninfer_sparse_moe_bench` on the sparse-MoE
prefill operator; the end-to-end figures at the end are an observation and do not isolate it.

## The observation

`sparse_moe_prefill_gather_kernel` materialises one 2048-wide BF16 row for **every (token, expert)
assignment**. At top-8 of 256 that is eight copies of every token: a 4096-token slice stores 32,768
rows of 4 KiB, **128 MiB**, and an 8192-token chunk stores that twice because the plan slices at
4096 - all of which `sparse_moe_prefill_q4_gate_up_kernel` then reads straight back through
`stage_inputs`.

Every one of those rows is a byte-exact copy of a column of the layer input `x`. The GEMM does not
need the copy, it needs to know **which token each packed column came from**. So the copy kernel can
become an index kernel that publishes only that map, and the staging loop can follow it into `x`:

```cpp
-                const int packed_col = begin + column_base + col;
-                const auto* src = gathered +
-                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kHidden + ...
+                cp_async_zfill<16, Cache::cg>(dst, src_row[i] + k0 + k8 * 8, col < cols ? 16 : 0);
```

A thread stages the same columns of every one of the `kHidden / kExpertBK` tiles, so it resolves the
rows those columns come from **once per route job**, into `src_row[]`, and the k-loop then walks
pointers. That keeps the map out of shared memory: **shared per block is identical to master's**, so
the `__launch_bounds__(ExpertWarps * 32, 3)` occupancy target is met exactly as before rather than
approached from underneath. The addressing is the only thing that changes: same codes, same scales,
same accumulation order, same `cp_async_zfill` predicate for the columns past the tail.

**The W8 routed path keeps the gather.** The change is registered only where the routed gate/up is
`Q4G64_F16S`, so the W8 codec is an untouched control that has to stand at 1.000 in the operator
bench, and a dense artifact is an untouched control end to end. Question 3 below is about whether
you want the other half; I have built and measured it, and the numbers are there.

## The interaction with `3a61ef3f`, and why it goes the right way

`fix(ops): separate sparse moe gather index lifetimes` split what used to be one buffer, because in
the gather all threads of an assignment block consume the tile-local rank while thread 0 publishes
the packed column into the same place. Your comment says exactly that.

The index kernel is **one thread per assignment**. It reads the rank from `local_rank` and writes
the map from that same thread, so the overlap the fix is about does not arise on this path at all.
The fix stays where it is and keeps protecting the gather, which is still there for W8.

## The workspace does not grow

The original form of this change allocated `packed_token` with its own `arena.alloc`, which is 256
KiB at chunk 8192 and would have made a change about saving traffic cost memory. It does not need
to. You had just shown where to put it:

```cpp
    out.tile_counts    = arena.alloc(DType::I32, {256, route_tiles}, 256);
    out.packed_index   = Tensor(out.tile_counts.data, DType::I32, {assignments});
+   out.packed_token   = Tensor(static_cast<std::int32_t*>(out.tile_counts.data) + assignments,
+                               DType::I32, {assignments});
```

Scan is the final consumer of `tile_counts`, and the arithmetic leaves room for both maps rather
than one: the buffer is `256 * ceil(T / 8) >= 32 * T` elements and each map is `8 * T`, so the two
together take **half** the dead prefix. **No allocation is added or removed and the arena sequence
is byte-identical to master's**, which makes `sparse_moe_prefill_workspace_bytes` unchanged by
construction rather than by measurement - and `ninfer_sparse_moe_bench` confirms it. It reports
`workspace_bytes` of 43,378,176 / 86,752,768 / 173,501,952 / 173,501,952 at
T = 1024 / 2048 / 4096 / 8192, identical on both arms at every width, and the engine's own
`gpu workspace peak` for the 35B artifact at 131,072 context is 835.06 MiB on both arms with the
same zero KV headroom.

`grouped_io` stays exactly as it is, and not because of the W8 gather. Your own comment says why:

```cpp
    //   gathered X BF16    <-> routed down output BF16
```

The routed down projection writes its output into that buffer and the reduce reads it back, so the
buffer does not leave even if nothing gathers into it any more. This saves **traffic, not
capacity**, and I would rather say that than quote a buffer that does not go away.

## Values are unchanged

`gathered[packed_col]` was written as `x[token]` by the gather, and `packed_token[packed_col]`
resolves to that same `token`, so the staged bytes are the same bytes. Columns past the tail read
the same fallback row the old code read and are zero-filled by the same `cp_async_zfill` predicate.
Greedy output is byte-identical: 24 generations, four rounds over six points, two artifacts and
two chunk widths. So are the 24 from an earlier campaign on the first form of this change.

## Measured effect

`ninfer_sparse_moe_bench --codec all --cache cold --execution eager --repeat 50`, eager because
prefill is not captured into a CUDA graph. Median of 50 after 5 warmups, cold L2 through a 256 MiB
flush. **Two independent passes**, so a row that moves can be told from a row that is noisy.

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

`w8-w8` is the same operator, the same benchmark and a codec this change does not reach; its eight
cells span **0.9965 to 1.0045** and that is the floor the Q4 rows should be read against.

The gain is the gather and nothing else, and the two independent measurements agree on that to the
microsecond. The trace below times the gather at 57.61 ms over 680 launches, 84.7 us each, and each
launch covers one 4096-token slice; at T = 8192 the operator runs two of them, 169 us, against a
measured saving of about 180 us. **The GEMM that stopped reading the copy barely moves** - the trace
puts it at -0.6%, inside what a single pass resolves.

**Trace**, `nsys profile -t cuda --cuda-graph-trace=node`, Qwen3.6-35B-A3B, 32K prompt, chunk 8192,
`--max-new 1`, **one pass per arm**:

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
what a single pass can resolve; the two other kernels that move, the routed down and the shared
gate/up, are the ones that no longer share the machine with 256 MiB of stores. Nothing else moves.

**End to end**, confirmation only. Arms alternate inside each round, every point its own process,
greedy, four rounds, otherwise idle box.

| model | chunk | prompt | per-round branch/master | prefill | decode |
| --- | --: | --: | --- | --: | --: |
| Qwen3.6-35B-A3B | 1024 | 4,357 | 1.0215 1.0220 1.0224 1.0210 | **+2.17%** | +0.01% |
| Qwen3.6-35B-A3B | 1024 | 16,441 | 1.0180 1.0191 1.0181 1.0192 | **+1.86%** | -0.01% |
| Qwen3.6-35B-A3B | 1024 | 33,031 | 1.0141 1.0154 1.0144 1.0149 | **+1.47%** | -0.01% |
| Qwen3.6-35B-A3B | 8192 | 16,441 | 1.0195 1.0163 1.0207 1.0173 | **+1.85%** | -0.01% |
| Qwen3.6-35B-A3B | 8192 | 33,031 | 1.0151 1.0143 1.0149 1.0143 | **+1.47%** | +0.02% |
| Qwen3.6-27B dense | 8192 | 16,441 | 0.9950 1.0040 0.9954 1.0045 | -0.03% | -0.01% |

The dense 27B row is the null control and behaves like one: four ratios straddling 1.000 and
averaging to -0.03%. It has no sparse MoE at all. I quote the paired per-round ratios rather than a
mean of means because the absolute rate drifts between rounds on both arms and the ratios do not.

## Resources

`cuobjdump --dump-resource-usage`, master against branch:

| instantiation | REG | SHARED | LOCAL | STACK |
| --- | --- | --- | --- | --- |
| `q4_gate_up<8, 64>` | 75 -> 80 | 33792 -> **33792** | 0 -> 0 | 0 -> 0 |
| `q4_gate_up<4, 32>` | 116 -> 124 | 25600 -> **25600** | 0 -> 0 | 0 -> 0 |
| `gather_kernel<0/1>` | 20, unchanged | 0 | 0 | 0 |
| `index_kernel<0/1>` | new, 16 | 0 | 0 | 0 |

The other 17 `sparse_moe_prefill` instantiations are untouched in all four columns.

Shared is the number that matters, because `__launch_bounds__(ExpertWarps * 32, 3)` asks for three
blocks per SM. The device reports `sharedMemPerMultiprocessor` 102400 and
`reservedSharedMemPerBlock` 1024, so master's `<8, 64>` sits at 33792 x 3 = 101376 with 1024 bytes
of headroom. **The branch sits at the same number**, because the map lives in registers rather than
in a shared tile: 80 x 256 x 3 = 61440 registers against 65536, local and spill and stack zero on
both sides.

I wrote the shared tile first, since it is the obvious form. It costs `ExpertBN * sizeof(int)` and
takes that headroom from 1024 bytes to 256, and it was also slower in both passes. Reading the map
inline in the staging loop is the smallest diff of the three and was slower still, by 0.4 to 1.4% on
every Q4 point, because the map and the 64-bit row multiply are then redone for each of the 32
k-tiles.

## Checks not run

- No `ncu` counters: `RmProfilingAdminOnly` is set on this host. Kernel-level claims come from the
  benchmarks, the trace and `cuobjdump`, not from hardware counters.
- **One `nsys` pass per arm.** The attribution table above is single-pass and should be read as
  attribution rather than as a measurement; the operator table is the measurement.
- `docs/performance.md` publishes prefill throughput for `qwen3_6_35b_a3b` at a 1024-token prefill
  chunk, which is a path this change is on, so those figures would move. I have not re-run that
  harness - it uses INT8 group-64 KV, prefix reuse, CUDA Graphs and five seeds through the serve
  corpus - and I am not proposing edits to the document.
- The adaptive small-T route takes the index kernel through the same launcher branch, but nothing
  in this campaign forces it. It is covered by `ctest` and by no benchmark here.
- Qwen3.8-27B NVFP4 takes no sparse-MoE prefill route and is not measured.
- Speculation is off in every run here, and every end-to-end run uses bf16 KV.

## Questions

1. Is this within scope, and is an external implementation appropriate? It is one commit in two
   files.
2. Is aliasing `packed_token` into the dead prefix of `tile_counts` the placement you want? It keeps
   the workspace byte-identical and it follows what `3a61ef3f` did, but it does put a second live
   map in a buffer whose lifetime you had just made explicit, and a plain `arena.alloc` costing 256
   KiB at chunk 8192 is the alternative.
3. Should the same treatment go to the W8 routed path? It is the same copy for the same reason. I
   left it out of this submission because taking it would remove the only untouched codec in the
   operator benchmark, but I did build it, because the answer changes the shape of the change rather
   than only its size: with both routed codecs staging from `x` the gather kernel has no caller left
   and the launcher stops branching on the codec at all.

   I built that arm and measured it on the same two passes. `w8-w8` reads:

| T | 1024 | 2048 | 4096 | 8192 |
| --- | --: | --: | --: | --: |
| pass 1 | 1.0406 | 1.0673 | 1.0741 | 1.0742 |
| pass 2 | 1.0443 | 1.0707 | 1.0737 | 1.0711 |

A larger relative gain than Q4 sees, against the same +/-0.45% floor. Say the word and it is the
same patch; the reason it is not in front of you is that taking it removes the only untouched codec
in the table above.
