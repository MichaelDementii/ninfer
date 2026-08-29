**Level of the claim: kernel.** The end-to-end figures at the end are an observation, not the
evidence.

## The observation

`sparse_moe_prefill_gather_kernel` materialises one 2048-wide BF16 row for **every (token, expert)
assignment**. At top-8 of 256 that is eight copies of every token: TBD_GATHER_MB of stores per
4096-token slice, which `sparse_moe_prefill_q4_gate_up_kernel` then reads straight back through
`stage_inputs`.

Every one of those rows is a byte-exact copy of a column of the layer input `x`. The GEMM does not
need the copy, it needs to know **which token each packed column came from**. So the copy kernel can
become an index kernel that publishes only that map, and the staging loop can follow it into `x`:

```cpp
-                const int packed_col = begin + column_base + col;
-                const auto* src = gathered +
-                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kHidden + ...
+                const auto* src = x + static_cast<std::int64_t>(Tok[col]) * kHidden + k0 + k8 * 8;
```

`Tok[ExpertBN]` is filled once per route job from the published map and read by every stage of the
k-loop. The addressing is the only thing that changes: same codes, same scales, same accumulation
order, same `cp_async_zfill` for the columns past the tail.

**The W8 routed path keeps the gather.** The change is registered only where the routed gate/up is
`Q4G64_F16S`, so the W8 codec is an untouched control that has to stand at 1.000 in the operator
bench, and a dense artifact is an untouched control end to end.

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
than one: `256 * ceil(T / 8) >= 32 * T` elements against `8 * T` per map, so the two together use a
quarter of the dead prefix. **No allocation is added or removed and the arena sequence is
byte-identical to master's**, which makes `sparse_moe_prefill_workspace_bytes` unchanged by
construction, not by measurement - and the reported peak confirms it: TBD_WORKSPACE.

`grouped_io` stays exactly as it is. It is a lifetime union - gathered X against the routed down
output - and both the W8 gather and the down path still use it, so this saves **traffic, not
capacity**, and I would rather say that than quote a buffer that does not leave.

## Values are unchanged

`gathered[packed_col]` was written as `x[token]` by the gather, and `Tok[col]` resolves to that same
`token`, so the staged bytes are the same bytes. Columns past the tail read the same fallback row
the old code read and are zero-filled by the same `cp_async_zfill` predicate. Greedy output is
byte-identical: TBD_GATE.

## Measured effect

TBD_MEASURED

## Resources

| kernel | master | branch |
|---|---|---|
| `q4_gate_up<4, 32>` | REG 116, SHARED 25 600 | REG 116, SHARED 25 728 |
| `q4_gate_up<8, 64>` | REG 75, SHARED 33 792 | REG 76, SHARED 34 048 |
| `gather_kernel<0/1>` | REG 20, SHARED 0 | unchanged, still used by W8 |
| `index_kernel<0/1>` | - | REG 16, SHARED 0 |

The shared growth is exactly `ExpertBN * sizeof(int)` - the `Tok` array and nothing else. TBD_OCCUPANCY

## Checks not run

TBD_LIMITS

## Questions

1. Is this within scope, and is an external implementation appropriate? It is one commit in two
   files.
2. Is aliasing `packed_token` into the dead prefix of `tile_counts` the placement you want? It keeps
   the workspace byte-identical and it follows what `3a61ef3f` did, but it does put a second live
   map in a buffer whose lifetime you had just made explicit, and a plain `arena.alloc` costing 256
   KiB at chunk 8192 is the alternative.
3. Should the same treatment go to the W8 routed path? It is the same copy for the same reason; I
   left it out because the W8 gate/up stages through a different codec loop and that is a second
   mechanism.
