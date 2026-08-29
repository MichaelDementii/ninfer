**Level of the claim: operator.** The end-to-end figures are confirmation. **This changes a public
workspace boundary** - `sparse_moe_prefill_workspace_bytes(max_tokens)` returns more for
`max_tokens > 4096` - so the cost is stated before the gain.

**You have already published this change's price.** When you closed #96 you wrote that the MoE
prefill workspace grows "from about 120 MiB at 1024 to 482 MiB at 4096 and 963 MiB at 8192", that
it comes out of KV capacity, and that this does not justify changing the default chunk. The 963 MiB
is the figure with this constant raised - on `master` the ceiling caps `capacity_tokens` at 4096, so
a 8192-token chunk asks for the 4096 figure. **I am not proposing to change the default chunk.** The
default stays at 1024 and gets the same plan, the same workspace and the same arithmetic, by the
shape of the expression rather than by measurement. What changes is only what a caller who has
already chosen a chunk wider than 4096 gets - and for that caller the price is your number,
482 -> 963 MiB.

## The observation

`resolve_sparse_moe_prefill_plan` slices the prefill chunk at a fixed 4096 tokens whatever the
caller asked for:

```cpp
inline constexpr std::int32_t kSparseMoePrefillSliceMax = 4096;
...
const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);  // workspace
const std::int32_t slice_tokens    = std::min(tokens,     kSparseMoePrefillSliceMax);  // execution
```

A caller that asks for an 8192-token prefill chunk therefore gets two independent 4096-token passes
through the whole routed MoE. Each pass routes, scans, and runs its own grouped GEMM, so **every
expert's column run is half as long as the chunk would allow**, and the length of that run is the
whole economy of the grouped route: a route job re-reads its expert's entire weight slab, and at
top-8 of 256 the average run is `tokens / 32`.

Nothing in the kernels requires 4096. The constant is a ceiling on the workspace, not on
correctness.

## The change

One line.

```diff
-inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 4096;
+inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 8192;
```

## Nothing at or below 4096 changes, and that is by construction

The constant is only ever read inside a `min()`. For any `tokens <= 4096` and any
`max_tokens <= 4096` both expressions return exactly what they returned before, so the default
1024-token chunk gets the same plan, the same workspace and the same arithmetic. That is not a
measurement, it is the shape of the expression - and the operator benchmark then confirms it,
because it is its own control: the rows at 1024, 2048 and 4096 have to read 1.000 with an unchanged
`workspace_bytes`, and only the 8192 row may move.

## What it costs

This is the half of the change that matters, and it is the objection you have raised before about
wide chunks: the workspace comes out of KV capacity.

TBD_COST

## What it gains

TBD_GAIN

## Values are unchanged

Slice width divides the token range and changes no reduction: the grouped GEMM accumulates over the
hidden dimension, and a wider slice adds columns rather than reordering a sum. Greedy output is
byte-identical: TBD_GATE.

A host-side constant should also move no device code, and it does not: TBD_SASS.

## Checks not run

TBD_LIMITS

## Questions

1. Is 8192 the right ceiling, or should it stop being a constant? The number is a bound on the
   workspace, not on the kernels, so the honest form might be to derive it from the largest prefill
   chunk the engine is configured for. I did not do that here because a 32768-token chunk would then
   ask for TBD_32K of MoE workspace, and where that line goes is your call rather than mine.
2. `docs/performance.md` publishes prefill throughput at a 1024-token chunk, which this change
   leaves untouched by construction. Do you want the document to say anything about wider chunks?
