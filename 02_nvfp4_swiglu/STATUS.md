# NVFP4 fused SwiGLU: measured state, 2026-08-29

Branch on the build server: `perf/nvfp4-fused-swiglu-every-width`. Rebased onto `1fc1cb76` after
review (was `3a61ef3f`); one commit, one file, +10/-4. None of the four commits between those bases
touches `src/ops/linear_swiglu`, and the figures below were taken on `3a61ef3f`. Cherry-picked from our `72219916`. The second NVFP4 commit,
`c5f7ace1` (walk the TMA grid in token groups), is a **different mechanism** and is deliberately not
in this branch.

Raw data: `/root/qual/gNV/` on the server. Driver `/root/nv_campaign.sh`, oracle run
`/root/nv_oracle.sh`, log `/root/nv_campaign.log`.

## What the change is

`resolve_route` registered the fused TMA SwiGLU at exactly `T == 1024`, although the kernel accepts
any `T % 256 == 0`. Every wider chunk therefore fell into `LinearW4A4Post`: a linear writing
`34816 × T` bf16 into the arena and a separate `silu_mul` reading it back.

## Operator, measured

`ninfer_nvfp4_linear_swiglu_bench --policy a4 --t-sweep 256,512,768,1024,1280,2048,4096,8192
--repeat 50`, both arms, same build directory.

| T | master µs | branch µs | speedup |
|---:|---:|---:|---:|
| 256 | 155.6 | 155.6 | **1.000** |
| 512 | 313.3 | 313.3 | **1.000** |
| 768 | 411.6 | 411.6 | **1.000** |
| 1024 | 417.8 | 417.8 | **1.000** |
| 1280 | 530.4 | 501.5 | **1.058** |
| 2048 | 837.7 | 767.7 | **1.091** |
| 4096 | 1681.4 | 1497.1 | **1.123** |
| 8192 | 3456.0 | 3086.1 | **1.120** |

The exact `1.000` cells below 1280 are not a noise floor: they are the same route running the same
kernel on both arms, so they are a built-in control that the change touches nothing the route table
already covered. The gain starts at 1280, the first multiple of 256 above 1024, which is precisely
what the commit registers.

## The workspace capacity contract, enumerated

The commit edits `nvfp4_linear_swiglu_workspace_capacity_bytes`, and workspace boundaries are
contract, so this was settled by enumeration rather than argument: the function was called on both
arms over a grid of 4560 `(min_tokens, max_tokens)` intervals and the results diffed
(`/root/ws_contract.sh`, results in `/root/scratch_ws/cap_{mst,nv}.txt`).

| | |
|---|---:|
| intervals probed | 4560 |
| where the branch asks for **more** | **0** |
| where the branch asks for less | 963 |
| where throw behaviour changed | 0 |

So the capacity never regresses anywhere, and the validation surface is untouched. The reductions
come in two classes:

- **952 intervals fall by exactly 72,512 bytes** - one token of the `34816 × T` bf16 intermediate
  (69,632) plus one token of the NVFP4 activation codes and scales (5120/2 + 5120/16 = 2,880), not
  padding. This is the `last_baseline` step-down: when `max_tokens` itself
  now resolves to the fused route, the baseline term is sized for `max_tokens - 1`.
- **11 intervals fall by the whole intermediate**, one per width in
  {1280, 1536, 2048, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 131072}. These are the collapsed
  intervals, `min_tokens == max_tokens`: the materialising route is then unreachable inside the
  interval and the buffer leaves entirely. At 8192 that is 594,018,304 → 23,592,960 bytes, a saving
  of 570,425,344 = **544 MiB**, and 34816 × 8192 × 2 = 570,425,344 exactly.

**Which one is the product?** The measured peak answers it without reading more code: if the engine
requested a collapsed interval at chunk 8192, the branch would have shed 544 MiB and the reported
peak would have moved. It did not - 1.19 GiB on both arms. So the engine requests a non-collapsed
interval, the real saving is the 72,512 bytes, and it is far below the granularity the run summary
prints.

## Correction to our own earlier notes

`ПЕРЕНОС_35B_на_27B_NVFP4_2026-08-26.md` and the 27 August analysis claim this frees "570 MB of
arena". **In the product, measured: it does not.** The GPU workspace peak is identical on both arms
— 152.57 MiB at `--prefill-chunk 1024` and 1.19 GiB at 8192 — see `/root/qual/gNV/ws_*.err`.

The 570 MB figure was not invented: it is exactly the collapsed-interval saving above, 570,425,344
bytes at width 8192. It is simply not the interval the engine requests, so it never materialises as
a product-level win. Writing "frees 570 MB" in the submission would be an unsupported resource
claim of precisely the kind that gets a PR closed; writing "the capacity never increases, falls by
72 KiB in the interval the engine uses, and would shed the whole intermediate on a collapsed
interval" is the same fact stated truthfully.

The reason is in `nvfp4_linear_swiglu_workspace_capacity_bytes` itself:

```cpp
std::int32_t last_baseline = max_tokens;
if (resolve_route(policy, last_baseline) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) { --last_baseline; }
if (last_baseline >= std::max(min_tokens, 49)) {
    maximum = std::max(maximum, baseline_workspace_bytes(last_baseline));
}
```

The capacity deliberately still covers the materialising route at the largest `T` that takes it. It
has to: `program_impl.h` sizes each chunk as `std::min(prefill_chunk, end - cursor)`, so the final
chunk of a prompt is a remainder and is generally not a multiple of 256. On a 33,031-token prompt at
chunk 8192 that is four fused chunks and a tail of 263.

**So the claim is traffic, not capacity, and it must be written that way.** Saying "frees 570 MB"
would be the kind of unsupported resource claim that gets a PR closed.

## End to end, Qwen3.8-27B NVFP4

Arms alternating inside each round, one process per point, greedy, four rounds.

| point | master t/s | branch t/s | delta | round spread | separated |
|---|---:|---:|---:|---:|---|
| chunk 8192, 33k prompt | 7191.4 | 7362.6 | **+2.38%** | 0.91% | yes |
| chunk 8192, 65k prompt | 5781.3 | 5894.8 | **+1.96%** | 0.70% | yes |
| **chunk 1024, 33k prompt** | 6801.3 | 6802.2 | **+0.01%** | 0.32% | no, correctly |

The 1024 row is the control that matters most here: the product default is untouched, measured
rather than argued. "Separated" means the slowest branch run beat the fastest master run.

## The output gate: the 32-token one is worthless, the real one passed

The 32-token runs write 40-byte outputs, and across every point there are only **two distinct
texts**. All 20 comparisons came back IDENTICAL and that means almost nothing — worse, it invites
the false conclusion that the two routes agree bit for bit. They are not required to: the fused
epilogue holds the f32 accumulator and rounds once, the materialising one rounds gate and up to
bf16 and multiplies the rounded values. **Do not ship the 32-token result as evidence.**

The gate that does carry weight: the **65,882-token** prompt (`niah_32k.json`), `--max-new 512`,
stopping on the stop token after 164 generated tokens, 203 bytes.

| chunk | verdict |
|---|---|
| 1024 | **IDENTICAL**, 203 bytes — expected, the route does not change here |
| 8192 | **IDENTICAL**, 203 bytes — the route does change here, and the greedy text still agrees |

State this carefully. It is one prompt, and greedy decoding absorbs small numeric differences at
the argmax, so it shows agreement rather than proving bit-identity — which the change does not
claim and cannot have. The operator residuals in the next section are the real numerical evidence.

## Numerical residuals, both arms, same FP64 oracle and criterion

`NINFER_OP_REPORT_STATS=1`, A4 criterion `rel_l2 <= 1.6e-1`, gross ratio is max-abs over the
gross limit, so under 1.0 passes.

| T | route on master | master rel_l2 (% of limit) | branch rel_l2 (% of limit) | master gross | branch gross |
|---:|---|---:|---:|---:|---:|
| 5 | fused small-T | 0.111 (69.4%) | 0.111 (69.4%) | 0.639 | 0.639 |
| 48 | fused small-T | 0.0836 (52.3%) | 0.0836 (52.3%) | 0.639 | 0.639 |
| 49 | materialising | 0.08358 (52.2%) | 0.08358 (52.2%) | 0.639 | 0.639 |
| 128 | materialising | 0.06621 (41.4%) | 0.06621 (41.4%) | 0.589 | 0.589 |
| 1024 | fused TMA | 0.05269 (32.9%) | 0.05269 (32.9%) | 0.373 | 0.373 |
| **1280** | materialising | 0.05159 (32.2%) | **0.05126 (32.0%)** | 0.336 | **0.307** |
| **2048** | materialising | 0.05011 (31.3%) | **0.04969 (31.1%)** | 0.286 | **0.261** |
| **4096** | materialising | 0.05234 (32.7%) | 0.05234 (32.7%) | 0.286 | **0.326** |

Two things to say honestly and neither is spin:

1. Every width the route table already covered reproduces to the last printed digit on both arms.
   That is the confinement claim at the numerical level, not just the timing level.
2. At the three re-routed widths the fused route's relative-L2 is equal or very slightly lower, and
   its gross ratio is lower at 1280 and 2048 but **higher at 4096 (0.326 against 0.286)**. So it is
   not "strictly more accurate" — say "equal or slightly better on relative-L2, mixed on the gross
   tail, and every cell far inside the criterion". For scale, the widths the project already ships
   sit at up to 0.639 of the same gross limit, so the new route stays well inside the envelope the
   project already accepts.

## Numerics

The shipped test `tests/ops/linear_swiglu/test_nvfp4.cpp` qualifies **both** routes against an FP64
oracle (`linear_swiglu_oracle_fp64`): `kA4Cases{5, 48, 49, 128, 1024}` puts 49 and 128 on the
materialising route and 1024 on the fused one. This change moves neither of those, so the shipped
verdicts are unchanged.

It does not cover 1280 and up, which is exactly what the change re-routes. Running the same test
with `kA4Cases` extended to `{5, 48, 49, 128, 1024, 1280, 2048, 4096}` on both arms — where master
takes the materialising path at the new widths and the branch takes the fused one — **both arms
pass** the A4 criterion. Per-width residuals are being collected with `NINFER_OP_REPORT_STATS=1`.

Worth quoting in the submission, from `linear_swiglu_test_common.cpp:29`, the maintainer's own
comment above the criterion:

> The criterion belongs to the activation-compute profile, not the weight storage format or a
> private materialized/fused implementation.

That is the frame: fused versus materialising is a private implementation choice, and the contract
is the A4 profile's criterion, which both routes meet.

## Still to do

- [ ] finish round 4 of the end-to-end and recompute all rows with spreads
- [ ] per-width oracle residuals, both arms, into a table
- [ ] decide whether extending `kA4Cases` should be part of the PR (it probably should: the change
      newly routes those widths and the repo asks for relevant shapes and routes to be exercised)
- [ ] ISSUE_BODY.md / PR_BODY.md / COMMIT_MSG.txt
- [ ] independent review rounds, as was done for `01_w8_rowsplit`
- [ ] this is the **second** submission; it must wait until the W8 Issue is answered, since the
      repo caps a contributor at two open items and the Issue-first rule applies to both

## Open design point found in review: `kTmaBlockM` is duplicated across translation units

The branch adds `constexpr std::int32_t kTmaBlockM = 256;` to
`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`, duplicating
`Nvfp4W4a4TmaSchedule<256,3,1>::kBlockM` from `nvfp4_linear_swiglu_w4a4_tma.cu` with no
`static_assert` tying them together.

Master was immune to this: 1024 is a multiple of 128, 256, 512 and 1024, so any retune of the
schedule stayed safe. With the predicate keyed on 256, a schedule moved to `kBlockM = 512` — which
the template permits — would send `T = 1280` to a launcher that throws at runtime, in prefill.

The fix is small and should probably be in the submission: put the constant in
`nvfp4_linear_swiglu_w4a4_tma_launch.h`, `static_assert` it against the schedule in the `.cu`, and
have `plan.cpp` use it. That makes the diff touch three files instead of one, which is why it is
recorded here as a decision rather than applied silently — but a duplicated magic number that
turns into a runtime throw is exactly the kind of thing the maintainer objects to.

## Two things the review changed after this file was first written

**ctest is now green.** One of the four commits between `3a61ef3f` and `1fc1cb76` is
`fd48e2fa test(cache): fix host restore output oracle`, which rewrites the oracle of
`test_engine_prefix_real.cpp` — the test that was failing for us and that we reported as **#105**.
He fixed the test's expectation, not the engine. On the rebased base, `ctest -j1` is
**94 tests, 0 failed, 1 skipped** on master and on both branches. Both submissions should say that
rather than carrying a "one pre-existing failure" paragraph.

**The lower bound of the predicate was never measured, and the claim about it was false.** The
submission said 1024 is the bound "because that is where the fused path starts winning". Nothing in
the campaign measured the fused route below 1024 — the 1.000 rows at 256/512/768 time the
*materialising* route on both arms. An independent probe says the fused route is faster there too,
and by more: roughly x1.12 at 256, x1.35 at 512, x1.28 at 768. Being verified independently now.

If it holds, the honest move is not to quietly lower the bound but to say so and offer it: the
change as it stands is conservative, the three largest per-width wins are left on the table, and
lowering the bound to `kTmaBlockM` is a one-token edit that re-routes widths the shipped test does
not cover on either route. That is a decision for him, and putting it in the Issue is exactly what
the Issue is for.
