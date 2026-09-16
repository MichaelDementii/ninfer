# What the NInfer maintainer accepts — rules distilled from every upstream PR and issue

Source: complete read of `github.com/Neroued/ninfer` — 107 pull requests (#1–#233) and 124 issues
(#3–#234) as of 2026-09-13, plus the current `AGENTS.md` and `CONTRIBUTING.md` on upstream master
(`d4929686`). Quotes are the maintainer's own words, verbatim, with the thread they came from.

This file is a working reference for preparing contributions that get merged. It is *not* a copy of
`AGENTS.md`; it records what actually decided outcomes in practice.

**Coverage:** all 107 PRs and 124 issues were retrieved and read. One thread, PR #224, could not be
rendered by the retrieval tool (its body contains a raw `<|im_start|>` token) and is represented only
by its list-page state (closed Sep 10, 2026). Everything else is first-hand.

---

## 0. Baseline facts that decide most outcomes

| Fact | Consequence for a contribution |
|---|---|
| Single owner, single reviewer, review is a limited resource | Reviewability is a first-class acceptance criterion, not politeness |
| One GPU target: RTX 5090, `sm_120a`, CUDA 13.1, Ubuntu 24.04 | Portability arguments carry no weight; other-GPU support is out of scope |
| Closed set of registered artifacts (`qwen3.6-27b` int/nvfp4, `qwen3.8-27b` int/nvfp4, `qwen3.6-35b-a3b` int) | "Generic model support" is an anti-feature |
| "support" has a specific meaning | *"'support' in ninfer means 'can run' + 'fully optimized'"* — issue #217 |
| No backward compatibility for project-owned surfaces | Leaving a fallback/alias in place is a defect, not caution |
| Codex (ChatGPT) reviews every PR automatically | Its P1 findings are effectively blocking; answer each one with a commit or an argument |
| Our fork is behind upstream | *"ok, remember do a rebase to master"* — issue #186. Every measurement must be on the base of the day |

---

## 1. Scope: what will be refused no matter how good the code is

These were refused with working code and, in several cases, with measurements attached.

- **Other GPU architectures / platforms.**
  - sm_89 (Ada): *"Supporting sm_89 would be difficult for me to maintain, so it's currently out of scope."* (#189)
  - RTX 3090: *"3090 is not the target here, try the 3090 fork"* (#18)
  - Native Windows (repeatedly, #13, #76, #82, #84, #233): *"native Windows support still requires
    platform-specific workarounds in core CUDA kernels… I don't want to carry that additional
    complexity and maintenance burden. Since we already provide Docker support, please use Docker on
    Windows instead."* (#13)
- **Offloading / CPU or NVMe tiers.** *"I think offloading for a dense model is useless. If there's
  something like Qwen3.8 122B A10B, I'll add support."* (#29). Asked to choose between a native
  cold tier, an external cache connector, and "out of scope", the answer was a single character:
  **"3"** (#98).
- **Repairing malformed model output.** *"ninfer intentionally does not repair malformed model
  output. It follows the declared tool call contract and json schema instead."* (#101);
  *"NInfer is keeping tool call parsing strict as intended."* (#10). The full doctrine is in the
  #159 closure (below) and #158.
- **Batch-invariant greedy decode.** *"Not in scope. Greedy does not mean batch invariant. And batch
  invariant limits optimizations."* (#80)
- **Unsupported hardware sizing requests** (5060 Ti 16 GB, etc.): *"not in scope"* (#95).
- **Complexity that a cheaper fix already made unnecessary.** On on-demand vision residency:
  *"the idea is very good. However, [fc5c483] reduced vision workspace size, so it no longer
  meaningfully squeezes kv capacity. I don't think the additional complexity is necessary for now."*
  (#74)

**Rule:** check scope *before* writing code. A design that requires a new execution platform, a new
memory tier, a new model family, or a loosening of an external contract needs agreement first —
`CONTRIBUTING.md` says so explicitly ("Discuss a change before implementation when it would revise
an established architecture, ownership, artifact, numerical, or external protocol contract").

---

## 2. Reviewability: one mechanism per PR

The single most expensive lesson in the whole history is PR #96 (`perf: 35B-A3B decode +13.2%,
prefill +16.8%, MTP3 +4.8%`), a 38-commit, 61-file stack that was closed unmerged. Verbatim:

> - "A 38-commit, 61-file performance stack is not reviewable or maintainable as one change.
>   Individually described commits do not make the combined state independently verifiable."
> - "Several performance ideas are worth evaluating independently. Aggregate end-to-end numbers over
>   dozens of changes contain too much noise and interaction to attribute a gain. **A new PR should
>   contain one mechanism, or one inseparable dependency chain, rebased on current master, with
>   production-shape operator benchmarks, numerical validation, and resource deltas. Cross-kernel
>   fusion and prefetch work should additionally measure the relevant producer-consumer or CUDA
>   Graph step. End-to-end measurements should remain the final confirmation, not the sole
>   evidence.**"
> - "Please do not bundle product-contract changes into performance work. Expanding MTP from 5 to 15
>   drafts changes the CLI domain, state/workspace bounds, and Graph profiles; it requires a separate
>   proposal and workload-level evaluation."
> - "If you want to continue, please open new, narrowly scoped PRs based on current master."

That instruction was followed and the resulting narrow PRs (#99, #106, #112, #140, #150, #191, #198,
and #203, #204, #205, #206) were all merged. This is the single highest-leverage rule in this document.

Also from the same thread: *"The default prefill chunk of 1024 is deliberate. The reported workspace
grows from about 120 MiB at 1024 to 482 MiB at 4096 and 963 MiB at 8192. That memory comes directly
out of the capacity available for KV, cached contexts, and active requests, so the larger-chunk gain
is not free and does not justify changing the default."* — a performance gain that spends a shared
budget is not a gain.

The Chinese-language closure of issue #132 says the same thing bluntly:
*"想自己加功能就 fork，想提 pr 就老老实实提 // 一下子提这么多大 pr 谁给你审核谁给你维护？CONTRIBUTING.md
写的是一点也不看啊？"* ("If you want your own features, fork. If you want to submit a PR, submit it
properly. Who is going to review and maintain this many huge PRs at once? Did you not read
CONTRIBUTING.md at all?")

---

## 3. PR description format — the explicit specification

From the #220 closure. This is the only place the maintainer spells out the description contract, so
treat it as the template:

> "Please also keep PR descriptions concise and focused on the final change:
> - Start with the existing contract being violated, or the concrete optimization being proposed.
> - Explain what changes and why it is needed.
> - Summarize relevant validation and limitations; **put detailed measurements and logs in a separate
>   report**.
> - Update the description when the implementation changes. **Remove superseded claims and revision
>   history.**
> - Avoid repeating conclusions or presenting an assumed invariant as an established requirement.
>
> **A reviewer should be able to identify the problem, the change, and the supporting evidence
> without reconstructing the investigation.**"

`CONTRIBUTING.md` adds the required content list: the concrete problem; the design and why it is
appropriate; the affected behavior or contract; the exact verification commands and summarized
results; workload/hardware/toolchain for any performance claim; any relevant check not run and its
implication; and, for an entirely AI-generated implementation, the exact model and version.

Practical shape that has worked:

```
Problem      one paragraph: the contract violated, or the kernel and the measured gap
Change       what the diff does, mechanism-level, and the ownership boundary it respects
Result       headline operator delta + headline end-to-end delta, one line each
Validation   oracle used, byte-identity or criterion, ctest result, format
Limits       where it does not apply, what was not measured, what could not be run
Report       link to docs/maintainer/<topic>.md for the full measurement tables
```

Do **not** paste the whole investigation, rebase archaeology, corrections of earlier revisions, or
retracted paragraphs into the PR body. Our own merged PRs that did this (#203, #205, #200) drew the
description complaint even when the engineering was accepted.

---

## 4. Code requirements that appear as review comments again and again

1. **Ownership and no ambient state.** From #96: *"Hidden mutable channels such as thread-local
   setters or module-global device state are also not acceptable ownership."* Pass what a kernel
   needs as an explicit parameter through the owning structure (this is why #206 carries
   `SparseMoeHints` in `SparseMoeDecodePlan` rather than a thread-local).
2. **No generality the product does not use.** From #99: *"The production layouts are only
   2048 + 2048 + 4096 and 2048 + 2048 + 6144, and your own measurements show that the split
   convolution itself pays a 3.2%–5.0% cost. Please use compile-time specializations for the
   registered geometries, with wrapper dispatch based on geometry, so the production route does not
   pay for unused generality."*
3. **Calibration is not policy, and an assertion is not evidence.** From #167: *"Please simplify the
   large calibration/assertion section… retain geometry/resource invariants and necessary coverage
   checks, and move the detailed measurement history to a maintained performance reference.
   **Assertions that a predicate keeps returning a particular answer do not establish that the
   selected route remains faster.** Express the applicable calibration domain and known bounds
   clearly."*
4. **Reuse the helpers that exist.** From #167: *"the PR adds `fp8_mbarrier_init`, `fp8_mbarrier_wait`
   … while its base already provides the corresponding functionality in `src/ops/common/mbarrier.cuh`
   … Please reuse the existing helpers so synchronization fixes and maintenance have one owner."*
5. **State a source-level precondition explicitly.** From #167 (512-byte swizzle alignment):
   *"This is a missing source-level correctness precondition, not a claim that the submitted binary
   has already produced incorrect output. A particular compiler layout can satisfy the stronger
   alignment incidentally."* Add the `static_assert`; do not rely on the layout you happened to get.
6. **Do not make another kernel's private behaviour a public constraint.** From #167:
   *"Please distinguish a measured performance reason for excluding these widths from a restriction
   imposed only to simplify `memcmp`. The long-term qualification criterion is the production Op
   against its independent mathematical oracle at the defined semantic boundaries. Bitwise comparison
   with the previous kernel is useful supporting evidence, but should not become an additional public
   numerical contract."*
7. **Generic Op names, model docs stay mathematical.** From #99: *"The model documents should not
   record the concrete function name or whether a private intermediate tensor is materialized; the
   model mathematics did not change. Please also keep the generic Op parameter names independent of
   the first caller (`out0/out1/out2` rather than `out_q/out_k/out_v`)."*
8. **Compute resources; do not discover them by trial and error, and do not swallow exceptions.**
   The full statement is the #2 closure, which is the clearest single description of the quality bar:

> "It contained an obvious infinite-loop condition: `context-fallback-step` may be set to zero, in
> which case `max_context` never changes after the memory check fails, so the `while (true)` loop
> retries forever. The decrement logic can also go below the configured minimum context.
> **More fundamentally, at startup, given the available device memory, weight footprint, KV dtype,
> and per-token memory requirement, the available context size can be calculated directly. It should
> not be discovered through an inefficient trial-and-error loop.** In an era when AI assistance is
> readily available, I do not understand how code like this can still be submitted.
> The second PR … wraps the entire model materialization and construction process in
> `catch (const std::exception&)`. As long as another fallback remains available, materialization
> failures, construction logic errors, and even exceptions thrown by the progress callback are all
> misclassified as OOM, after which the code silently changes the KV dtype or context size and
> retries. **This masks the actual failure.** … the selected INT8 dtype exists only in a local
> variable. `Engine::options()` and the server configuration may still report BF16 … **This directly
> contradicts the PR description's claim** …
> I am interested in high-quality PRs, but **I am not willing to take responsibility for checking
> basic logic, redesigning the core approach, and fixing obvious errors on behalf of the
> contributor.**"

9. **Kernel/operator work is partly reserved.** *"Thanks for the PR, but I'd prefer to handle this
   kind of operator/kernel work myself."* (#6). In practice, kernel PRs *are* accepted — but only in
   the narrow, fully-qualified form described in §2 and §6.

---

## 5. Numerical validation rules

- Each floating-point Op is qualified against an **independent naive FP32/FP64 oracle**, at real
  model shapes and route boundaries; exact transforms/codecs against an exact oracle; packed inputs
  decoded independently with their stored scales (`AGENTS.md`, `docs/maintainer/op-development.md`).
- **Do not qualify a kernel against another kernel.** Bit-identity with the route you replaced is
  supporting evidence only (#167, #226).
- **Contracts state semantics, not implementation.** Codex P1 on #226, accepted by the author:
  a contract clause must state the formula, the eps domain, the BF16 boundaries and the FP64-oracle
  criterion — not freeze a reduction topology or promise byte-parity with a composed path.
- **Private intermediates are private.** The #220 closure is the canonical statement:

> "This is not a numerical requirement: **the Op contract explicitly allows private convolution
> intermediates to remain unrounded.** Exact equivalence is required only for ReplaySSM record/fold
> versus the corresponding full computation, which existing tests already cover.
> The reported restart differences do not violate that contract and do not justify introducing
> additional rounding."

  Same point in #96: *"The proposed GDN BF16 carry … is not an upstream correctness bug: that
  accumulator is private arithmetic, not a required BF16 state boundary."*
- **Precision experiments the maintainer has already run and rejected** — do not re-propose without
  new evidence: *"In attention, PV is much more precision sensitive than QK, even use bf16 PV with
  fp32 acc is worse than fp16 PV with fp32 acc. So using fp16 acc is no an option."* (#27)
- **Show the test can fail.** Sabotage/strength controls (deliberately damaged builds that the gate
  must detect) are the norm in accepted PRs, and Codex asks for them when missing (#195: a test that
  would still pass with the feature removed is not coverage).

---

## 6. Performance methodology — the actual bar

This is where our PRs were deferred or rejected, so it deserves the most attention.

### 6.1 Measure at the level you claim

`AGENTS.md`: *"Measure performance at the claimed scope. An Op microbenchmark establishes an Op
result, not an end-to-end improvement."* And #96: end-to-end is *"the final confirmation, not the
sole evidence"*.

### 6.2 Establish causality with a null-controlled, same-process experiment

The deferral of PR #201 defines the protocol by reference to the one used against PR #200:

> "for the feature Linear shape N=5120, K=25600, two opposite-order paired comparisons gave these
> candidate Op-time changes: T=60 → +1.037% / +0.475%; T=64 → +0.984% / 0.000% … **These small
> effects have not been causally established with the same-process/null-controlled experiment used
> for #200.** They are an unresolved acceptance concern, not a confirmed regression of that strength.
> Please qualify the BM=16 feature-route weight/scale policy with a focused controlled comparison
> around T=57/60/64, retaining the positive T=129 result as a check. If the residual cost is real,
> adjust the per-schedule policy or demonstrate the workload-level benefit that justifies accepting
> it. **Lack of an end-to-end gain alone is not the reason for deferral; the unresolved cost on the
> changed feature route is.** No broad profiling campaign is needed to address this question."

The maintainer's own protocol, from the #200 rejection, is the reference implementation:

> "The controlled experiment captured all three graphs in one process over shared inputs/buffers and
> interleaved replay through all six arm orders: 300 samples per arm after 20 warmups, with a 256 MiB
> cold-cache flush before each call. Three separate-binary comparisons also showed +1.613%, +1.613%,
> and +1.587%."

Elements to reproduce: **all arms compiled into one binary and captured in one process**, shared
inputs/buffers, **interleaved replay through every arm order**, warmups then hundreds of samples,
cold-cache flush per call, plus an **identical-baseline control arm** that must read 0.000%.
Codex enforces the same rule from `op-development.md`: *"the complete candidate-by-extent matrix [must]
be compiled together and collected in one run rather than produced by repeated edits and rebuilds"*
(#200 P2).

### 6.3 A regression on any supported route blocks the merge

PR #200 was rejected on a single cell, in the presence of large wins elsewhere:

> "I do not recommend merging this version because the Q6 Rows2 extension through T=17 regresses a
> supported routing case. … Baseline Rows4 127.008 us / Candidate Rows2 129.056 us / Identical
> baseline control 127.008 us. This is a 1.6125% Op-time regression, with zero difference between the
> identical baseline controls. … The rest of the change has useful results … No correctness or
> implementation-quality blocker was found. **Please requalify the Q6 upper boundary, including a
> cutoff of 16, or use a justified selection that avoids the T=17 regression.** … This is an Op-level
> performance finding, not a mathematical bug or a claim of 1.6% whole-model slowdown."

Corollary: sweep **every reachable width/route your change touches**, including the ones your headline
workload never hits. Codex found the same class of defect in #199 (the shared helper regressed the
T=1 nine-warp decode kernel by 6.1–6.6% while the MTP3 headline improved).

### 6.4 Report a roofline

The one piece of unsolicited methodological advice given as praise, on merged PR #112:

> "Great work. One suggestion for future performance reports: besides the speedup over the previous
> implementation, **a roofline analysis, e.g. achieved memory bandwidth and tensor core peak relative
> to the hardware limits** would be very useful. It helps clarify how much optimization headroom
> remains."

Measure the ceilings on the card you are using and cite them (the corpus uses 1689.4 GB/s read,
1503.4 GB/s DRAM copy, 253.4 TFLOP/s `mma_bf16` f32-acc, 2003.9 TFLOP/s NVFP4 MMA). Expect the
numbers to be checked: *"check the RTX5090 spec for the number, I remeber it's 1736 TFLOPS for
nvfp4"* (#160).

### 6.5 Claim only what the counters support

Codex P3 on #200, unchallenged by the maintainer: with `dram__*` counters unavailable, a flat
`lts__t_bytes.sum` establishes flat **L2** traffic only, not flat DRAM traffic — *"either collect a
direct DRAM/miss counter or limit the comment to the observed L1 and L2 counters."*

### 6.6 Re-measure on the base of the day, and expect interaction

PR #202 was withdrawn because two other L2 optimizations merged in between and inverted its sign
(+0.79% → −0.22% decode, −1.04% prefill). L2 capacity, shared-memory budget and occupancy are
*shared budgets*: an independently-validated win can be cancelled by another merged win. Always
re-measure immediately before asking for review, and say which base the numbers are from.

### 6.7 Thresholds are machine-dependent

The #200 post-mortem showed the Q6 crossover moving with SM clock (T=17 loses on both a 1975 MHz and
a 2810 MHz machine; the Q5/Q6 crossovers differ 17/19 and 11/16 between them). A tuned constant must
be justified on the binding (slowest-clock) case, or kept conservative.

---

## 7. AI-generated contributions

`CONTRIBUTING.md`: AI-assisted work is welcome, the contributor remains fully responsible for every
line, and *"if an implementation is generated entirely by AI, it will only be considered when produced
by a current state-of-the-art coding model. The exact model and version must be disclosed in the pull
request."* Disclosure alone is not enough — *"Generated code that has not received these basic
checks is not ready for submission."*

Older merged PRs in this corpus carry a `Co-Authored-By: Claude Opus 5` trailer and a "Generated
with Claude Code" line. **Do not copy them.** The maintainer asked for both to go on 2026-09-16;
12.8 has his words and what it means for a submission.

---

## 8. How closure works, and how the maintainer behaves

- A closed PR is a **threshold judgement**, not an invitation to debate: *"Closing such a pull request
  is a determination that it does not meet the project's review threshold. It does not create an
  obligation for the maintainer to debug, repair, or complete the contribution."* (`CONTRIBUTING.md`)
- He frequently **implements the fix himself** and closes the PR/issue with a commit link
  ("superseded by …", "fixed in …", "implemented in …": #24, #35, #45, #55, #71, #79, #88, #108).
  Getting the *problem* in front of him is often worth more than getting the patch merged — #14's
  investigation was praised and re-implemented as #15; #176's report produced the broader
  `d4929686` planner rework.
- He is fast and informal on bugs with a clean reproduction (*"shit, I'll take a look"* #20;
  *"Thanks for the clear report and reproduction. This was a real bug. Fixed in …"* #66), and short
  with low-effort reports (*"at least you should check --help and docs instead of open an issue"* #30;
  *"what's this"* #209).
- He asks for reproducibility before accepting a quality claim: *"I also don't have your prompts,
  schemas, or benchmark code, so I can't reproduce this result. If you think there's a quality
  regression, please provide a few reproducible examples…"* (#38)
- He can be persuaded by evidence — the tool-call parser was loosened after a user pushed back with a
  concrete case (*"I'll loose the parser rules."* → commit `3b50962`, #138) — but not by comparison
  to other engines' leniency (*"…At least… verify what are you saying and why. sglang has the loosest
  tool call parser among vllm, sglang and llama.cpp."*).
- Duplicates get merged into one thread rather than answered twice (#125, #153).

---

## 9. Outcome table — what actually happened to comparable work

| PR | Nature | Outcome | Deciding factor |
|---|---|---|---|
| #99, #106, #112, #140, #150, #191, #198, #203, #204, #205, #206 | one mechanism, kernel or host, full qualification | **merged** | narrow scope + oracle + operator *and* end-to-end + resource census |
| #115 | test-only, covers a newly reachable width | **merged** | closes a real hole; *": ) I noticed it and I thought it was fine. But nice work."* |
| #96 | 38-commit perf stack | closed | not reviewable; bundled contract changes; aggregate-only attribution |
| #200 | threshold retune | rejected | 1.61% Op regression on a supported routing case |
| #201 | cache-policy retune | deferred | +0.5–1.0% residual on the changed route not causally established |
| #202 | L2 persistence for LA state | closed (by author) | sign flipped on the current base after #203/#206 merged |
| #220 | rounding a private intermediate | closed | not a contract violation; description not focused |
| #159, #101, #10 | tool-call leniency | closed | conflicts with the atomic tool-call contract |
| #1, #2 | memory-adaptive startup | closed | trial-and-error resource discovery, blanket catch, false description |
| #13, #76, #82, #84 | native Windows | closed | maintenance burden in core CUDA; use Docker |
| #52 | raise MTP graph allowance | closed | *"caused by instability in the CUDA driver / graph capture … there is no artifact-specific limit or special case here"* |
| #116 | sampling presets + clamp | closed | defaults already match official recommendation; clamp changes documented semantics |

---

## 10. Pre-flight checklist

Before writing code:

- [ ] Is the change inside the current product contract (one GPU, registered artifacts, no new tier)?
- [ ] Does it revise an architecture/ownership/artifact/numerical/protocol contract? → open an issue first.
- [ ] Is it **one** mechanism, or one inseparable dependency chain?

Before opening the PR:

- [ ] Rebased on current upstream master; all numbers re-measured on that base.
- [ ] Operator benchmark at production shapes, **plus** the producer-consumer / CUDA-graph step for
      fusion or prefetch work, **plus** end-to-end as final confirmation.
- [ ] Every reachable width and route of the touched kernels swept, not just the headline case.
- [ ] Null control arm (identical baseline) reported and reading ~0.000%.
- [ ] All arms in one binary, one process, interleaved through every order, cold-cache flush, warmups
      + hundreds of samples.
- [ ] Roofline: achieved GB/s and TFLOP/s against ceilings measured on this card.
- [ ] Numerical qualification against an independent FP32/FP64 (or exact) oracle at real shapes and
      route boundaries; byte-identity claims backed by a gate proven able to fail.
- [ ] `ctest` on both arms from one build directory; `clang-format` clean; register/shared/spill
      census for every changed kernel body.
- [ ] Contract text states semantics (formula, domains, boundaries, criterion) — not topology or
      parity with the old kernel.
- [ ] No thread-local/global mutable channel; everything passed explicitly through the owning struct.
- [ ] No generality beyond registered geometries; superseded paths deleted, not kept as fallbacks.
- [ ] Description rewritten to the §3 shape; revision history and superseded claims removed; detailed
      tables moved to `docs/maintainer/<topic>.md`.
- [ ] Limits section: what is not covered, what was not measured, what could not be run.
- [ ] If entirely AI-generated: exact model and version disclosed.

After Codex reviews:

- [ ] Every P1 answered with a commit or a substantiated argument; every P2 answered explicitly.
- [ ] Description updated to match the final commit, not the first one.

---

## 11. Our own defects — the checks that caught them, added 2026-09-14

Everything in this section comes from preparing #238 and #239 and from two independent adversarial
reviews of those diffs. Each item is here because we actually shipped or nearly shipped the mistake.
**Read this section before every submission, not only §10.**

### 11.1 Issue first — the rule we ignored for the whole campaign

`CONTRIBUTING.md`, *Start with an Issue*, verbatim:

> Every bug, performance opportunity, feature request, protocol change, and architecture proposal
> **must begin with an Issue before implementation starts.** … Wait for the maintainer to confirm the
> scope and implementation direction **before investing in a pull request**. … **A pull request
> without a linked, confirmed Issue may be closed without detailed review.**

We opened PRs cold, every time. It is not the sole cause of the backlog — #112, #150, #193, #198,
#203, #204, #205 were merged with no Issue at all, while #191 (Issue #190), #206 (Issue #68) and
#106/#140 (Issues #117/#118) had one — but it is a free reason to skip a PR, and skipping is what is
happening.

- New mechanism → short Issue first, wait for direction.
- Work already finished → one line in the body, the shape that worked in #99: *"the direction was
  established in your review rather than in an Issue; if you would rather it be re-filed as an Issue
  first, say so."*
- Evidence on his bandwidth: Issue #186 asked, at length, in what order to send fourteen finished
  changes. The entire answer was **"ok, remember do a rebase to master"**. Short questions get
  answers; long ones get one line.

### 11.2 Every number in the body must belong to the branch head

Check before submitting, not after. Three of our open PRs failed this at once:

| PR | the body said | the head did |
|---|---|---|
| #160 | operator −19.4 / −13.5 / −16.6 %, prefill +5.99 % | −8.4 / −7.6 / −9.9 % after our own merged #204 took part of the gain |
| #200 | "a single threshold of 19", Q6 moved to 17 | Q5 5 → 17, Q6 untouched |
| #201 | "`cg` by default, `ca` on the two BM=16 schedules", 3 files | default unchanged, one schedule opts in, 2 files |
| #194 | the pre-fix formula, numbers from `ad0f3d38` | the folded form, numbers from `b88c0f6` |

The correction always existed — as a comment further down the thread. **He reads the description, not
the thread.** A body that contradicts its own diff is the stated reason PR #2 was refused: *"This
directly contradicts the PR description's claim."*

### 11.3 `clang-format` clean is not the same as formatted

`clang-format --output-replacements-xml` reported **zero** replacements on a file where we had just
written two 101-character lines against `ColumnLimit: 100`. It cannot always rewrap, so it stays
silent. Check both:

```sh
clang-format --output-replacements-xml FILE | grep -c '<replacement '
awk 'length > 100 {print FILENAME":"NR" = "length}' FILE
```

and compare the second against master, so pre-existing long lines are not mistaken for ours.

### 11.4 Three kinds of sentence that must never enter a body unmeasured

All three were caught in our own drafts:

1. **A share.** "this kernel is about 5 % of prefill kernel time" — it was **17.6 %**. Compute it from
   the profile, do not estimate it.
2. **A mechanism.** "the gain falls with T because the epilogue is computed once per output element" —
   both the epilogue and the MMA grow linearly in T, so the explanation was wrong. If the mechanism
   was not established, write the observation and say it is not explained.
3. **An adjective standing in for arithmetic.** "four orders of magnitude below the bf16 quantum" — the
   real ratio is ~3.1 orders. Give the two numbers instead; there is then nothing to argue with.

### 11.5 A small sample is not a bound

"0 of 2001 sampled points round to a different bf16" became, at 40 million points, **1 in 31 128**.
When the claim is about a rate, measure the rate. Expected flips are about `d / 2^-9` per output for a
relative perturbation `d`, so a 2001-point sample cannot see a 1-in-30 000 effect.

### 11.6 Compare against what master does, not against the ideal

The strongest form of a precision claim is not "we are close to the mathematics" but "we are this much
further from it than master already is". For #239: master's own float `silu` disagrees with the
mathematics once in 41 237 outputs and the new form once in 31 128 — one extra differing output per
130 000. The same reframing turned a Codex P2 into a refutation: the zeroing it warned about is
master's behaviour, and the change removes it.

### 11.7 The identical-baseline arm is mandatory on the local card

The card is shared with the Windows desktop through WDDM and a run can be preempted outright. Running
the master binary a second time under another label as a third arm showed nulls of **+1490 %, −78 %,
+86 %** — 36 of 150 passes on the first sweep. Without it, a quarter of the passes would have entered
the result as signal.

- Drop a pass for a cell when its two identical binaries disagree by more than 1 %, and report how
  many were dropped.
- Three passes are not enough after dropping; take six or seven.
- `nvidia-smi -lgc` is refused under WSL, so clocks cannot be pinned. Repetition is the only defence.
- Wall clock on this machine has a ±2 % null on prefill throughput. Below that, measure kernel time.

### 11.8 Answer a bot finding on its premise, not only on its request

Codex P2 on #239 asked for a test guarding against a gate near −90 collapsing to zero. The premise was
inverted: **master** returns a wrong zero there 31 379 times in 200 001 sampled points, and the change
returns one nowhere. Answer with the measurement, then offer the test rather than building a contrived
input — `AGENTS.md` asks for *represented* public inputs and warns against tests that mirror the
implementation. Then ask for a re-review explicitly.

### 11.9 Run an adversarial review before he sees it

Both PRs went through an independent reviewer briefed to find defects before Codex and before a tired
human. On #238 it found four P2s, **two of which were the commit message disagreeing with the diff**.
On #239 it found nine items, three of which were our own unmeasured sentences. None of that should
reach the upstream thread: once a thread starts, he reads the correspondence instead of the change.

### 11.10 State the cost of the change, not only its gain

A layout change that speeds up the reader usually slows the writer. #238 measured both: the TMA GEMMs
lose 47 ms and the quantizer gains 0.6 ms, so the trade is stated as a trade. A body that reports only
the favourable side invites the reviewer to find the other one.

### 11.11 A width added to a test proves nothing until the harness is sized for it

Adding T=1025 to `tests/ops/linear_add/test_nvfp4.cpp` produced "non-finite at index 6", which reads
exactly like a kernel defect on the new ragged path. It was not: the harness sized its host
activation and residual from a literal 1024 while the invocation list is what drives them, so the
width read past the end of the buffer. The symptom is indistinguishable from the defect you are
looking for, and the way to tell them apart is the same decisive experiment as everywhere else -
turn the route off and see whether it still fails. It did, so the fault was in the harness.

Before reporting a failure on a newly added shape, check every bound in the harness that was written
as a literal rather than derived from the case list.

### 11.12 A test at a new threshold must be able to fail

The first version of the 4/4 floor test added T=768, the floor itself. 768 is three whole M tiles and
therefore configuration-identical to the 1024 case already in the list: the test passes whether the
floor is two tiles, three tiles, or 769. It could not fail, so it was not evidence.

The pair 767 / 769 replaced it - the widest width the old route still owns and the narrowest the new
one admits. Straddle the boundary; a single case sitting on it is decoration. Ask of every added case:
name the wrong value of the constant this case would catch.

### 11.13 A constant shared by several call sites must be measured through each of them

The route floor is read by four Ops with different output policies - `linear`, `attn_input_proj`
(one row block scattered to four destinations), `gdn_input_proj` (two), `linear_add` (residual read
per token, in place). The first sweep measured one of them. Running the other three did not change
the decision, but it is what turned the single regressing cell at 640 from an anomaly of one
benchmark into a reproduction on a second instrument.

If the constant is shared, the sweep has to be too, or the body has to say plainly which call sites
were not measured.

### 11.14 Reviewing the diff and reviewing the body are two different jobs

Preparing the route floor, the first adversarial pass was given the diff. It returned eight findings,
all about the code, and none about a number. The second pass was given the diff, the body, and the
raw measurement record together, with one instruction: check that every figure in the body appears in
the record with the same value. It returned five more findings of the first rank, and all five were
in the text:

* a superlative in a code comment that the comment two lines below it falsified;
* a threshold comment that documented a value one greater than the constant it sat on;
* "conservative by one and a half tiles" where the gap is half a tile - a number that had never been
  computed, only felt;
* "all four Ops read 0.00 %" where the control table holds four bench cells over three Ops;
* "seven of 49 cells" where the grid is fifty, the missing cell unnamed - and it turned out to sit at
  the floor itself, on a geometry the claim depends on, having lost every pass to the null gate.

None of these is reachable by reading the diff, and none would have been caught by re-reading the body
alone: each needs the body and the raw file side by side. Make it a separate pass, give the reviewer
both, and state the instruction as arithmetic rather than as judgement.

The corollary about the missing cell is worth its own line: **a cell the null gate emptied is not a
cell that agrees with you.** Say which it was and how it was covered, or the first reader who counts
the grid will find the hole.

### 11.15 A green `ctest` with no test count is not a pass

Configuring a fresh worktree without `-DBUILD_TESTING=ON` produces a build with no registered tests.
`ctest` then prints `No tests were found!!!` and **exits 0**. A script that greps for failures finds
none, and the pre-flight line "ctest clean" gets written on the strength of a suite that never ran.

Always record the count, never the verdict: "115/115", not "passed". A claim with no denominator is
the shape this failure hides in. And when a count changes between branches - 114 on one base, 115 on
another - that is information about the base, not noise to round away.

### 11.16 An output-comparison gate must be checked for emptiness before it is believed

The witness for the Rows2 window compared generated completions between two binaries and reported six
of six identical. The files were **one byte each**: the CLI writes its completion to stderr, and the
script captured stdout. Two empty files compare equal, and the gate reports success in exactly the
voice it uses when it has worked.

This is the same defect as a gate that runs at a token count the change does not reach - and that one
had already happened on the same PR. Print the size of what was compared, next to the verdict, every
time. A witness that cannot say how many bytes it examined has not examined anything.

### 11.17 Compute the roofline denominator at the clock you measured at

A body reported 90.6 % of the bf16 tensor peak by dividing a figure measured at 2770 MHz by the peak
at the 2407 MHz nameplate clock. The numerator had been re-measured on a faster session and the
denominator had not, which inflates the efficiency by about a seventh and reads as overstating how
close the kernel already runs to the machine.

Two rules follow. Use the ceiling the maintainer states (1689.4 GB/s read on this part), not the
constant the benchmark prints in its own header - those are the tool's defaults, not a measurement.
And recompute the peak at the SM clock actually recorded for the cells reported: 170 x 512 x f_SM for
dense bf16. If the clock was not recorded per cell, the roofline share cannot be stated at all.

### 11.18 A precondition check needs its premise verified against the ISA, not assumed

A submission added a runtime check rejecting a `cg` opt-in whose k was not a multiple of 8, on the
stated grounds that `cp.async.cg` needs a 16 B-aligned source while the predicated path has no such
guarantee. Reading `memory.cuh` settles it: both branches issue a 16-byte `cp.async` when the copy is
16 bytes, and the natural-alignment requirement is on the copy size, not on the cache modifier. So the
invariant was already master's, the check was unreachable, and the permanent comment stated the wrong
rule about the hardware.

Removing it made the diff smaller, took a `throw` off the launch path, and removed a claim a reviewer
who knows the ISA would have corrected in public. Before adding a guard, establish that the condition
can occur; a guard justified by a misread of the ISA is worse than no guard, because it teaches the
next reader the misreading.

### 11.19 A SASS census compares bodies, not names, and not raw columns

The claim "adding a defaulted template parameter changes exactly one instantiation" is checkable in
the object file, and it is worth checking - but `cuobjdump -sass` lays two traps for that comparison.

**Names all change.** Adding a parameter changes the schedule type, so every mangled kernel name
differs between the two builds even where the code is identical. Matching functions by name reports
that nothing corresponds to anything.

**Columns all change too.** `cuobjdump` right-aligns its hex-encoding comments to the longest symbol
in the file. A longer mangled name shifts that column in every line, so a naive text diff reports all
thirty bodies as different when only whitespace moved.

Compare bodies by content, with `/* ... */` address and encoding comments stripped and runs of
whitespace collapsed, and count multiset differences rather than pairing by name. Done that way the
census gave what the claim predicted: 30 bodies and 454 `LDGSTS` on each side, exactly one body
differing, its 15 `LDGSTS` gaining `BYPASS`. Done naively it gave "all 30 differ", which is the kind
of result that gets a true claim deleted from a body for being unsupportable.

## 12. What changed on 2026-09-15, and what it invalidates

Master moved 21 commits in one night. Three of them change what a submission has to look like, and
one of them contradicts a rule written here the day before.

### 12.1 There is now a PR template, and it is the required shape

`.github/pull_request_template.md`: **Problem and scope** (with a `Related Issue:` field),
**Implementation**, **Verification**. That supersedes the Problem / Change / Result / Validation /
Limits shape inferred from the #220 closure. The `Related Issue` field makes the issue-first rule a
form field rather than a paragraph in CONTRIBUTING, so a blank there is now visible at a glance.

CONTRIBUTING gained a table of what evidence each kind of change owes. For performance tuning it is
"correctness evidence and a final performance report", and for Linear the report has a prescribed
format.

### 12.2 Retained performance reports must not contain speedups

`docs/maintainer/linear-tuning.md` §4, verbatim: retained reports "omit old-implementation
comparisons, speedup ratios, aggregate improvement scores, and candidate-search history. Candidate
measurements remain working evidence for dispatch decisions; the report presents the resulting
implementation."

Every body we have written is built the other way round - a table of percentages against master.
That framing is now explicitly not what a report is. The comparison still belongs somewhere: it is
the argument in **Problem and scope** for why the change is worth making. What goes in
**Verification** is the absolute performance of the final implementation: latency, logical GB/s,
bandwidth utilization, useful TFLOP/s and Tensor Core utilization at the priority points and both
bulk anchors, with the peak denominators named.

The priority points are fixed: the hot interval is `1 <= T <= 128` with 1, 4 and 8 emphasized, and
`T=512` and `T=1024` reported separately so that an average cannot hide a regression at one of them.

### 12.3 The roofline denominators are the bench's constants, not a recomputed peak

This overrides 11.17. `linear_bench` defines `kRtx5090DramGBs = 1792.0`,
`kRtx5090SustainedReadGBs = 1674.5`, `kRtx5090Bf16Fp32AccumulateTFLOPs = 209.5`,
`kRtx5090Fp8Fp16AccumulateTFLOPs = 838.0` and `kRtx5090Fp8Fp32AccumulateTFLOPs = 419.0`, and the Q4
worked example divides by exactly those, naming 1792 for the nominal reference and 1674.5 as the
separately labelled sustained-read reference. So quote the bench's constants and label which is
which; do not substitute a peak recomputed at the measured clock, and do not use 1689.4.

What 11.17 still gets right is narrower and worth keeping: do not silently mix a numerator measured
in one session with a denominator from another, and do not quote a utilization figure at all when
the denominator is not defined for that path.

**The NVFP4 path has no defined peak.** The bench carries tensor peaks for fp8 and bf16 only, and
its `TC_profile` and `TC_%` columns print a dash for NVFP4 rows. Measured logical throughput on the
W4A4 TMA route already exceeds every constant the bench defines, so no Tensor Core utilization can
be stated for it without a number nobody has established. Report the bandwidth columns, say plainly
that the bench defines no NVFP4 tensor peak, and leave the ratio out.

### 12.4 A shared threshold can stop being shared

The W4A4 TMA floor used to be `tokens >= 1024 && tokens % 256 == 0` everywhere. After
`refactor(ops): organize nvfp4 execution by matrix shape` it lives in each shape's own `select_a4`,
and `perf(ops): tune nvfp4 34816x5120 linear routes` then lowered **that one shape** to 256 while
the other four stayed at 1024.

Two consequences. A change that centralizes such a predicate is wrong the moment one call site
diverges, so a port must re-read every copy rather than assume they still agree. And a proposal to
move the threshold has to be per shape now, against whatever that shape currently uses - our own
floor work was measured when all five agreed, and one of the five has since moved past the value we
were going to propose.

### 12.5 A comparison arm must be rebuilt after anything touches its tree

Comparing two arms on identical test widths means putting the same test sources in both. Doing that
by copying the files across, running, and then restoring them with `git checkout -- tests/` leaves
the restored arm's **binaries** built from the copied sources. The next comparison then runs one arm
with extra widths and the other without.

The symptom is not a value mismatch. It is a mismatch in the **number** of records - `diff` reports
deleted lines rather than changed ones - and it appeared on two Ops that the change provably does not
touch, which is exactly where a real defect would have been alarming. Rebuilding the restored arm
made both read IDENTICAL.

Two habits follow. Rebuild any arm whose tree was touched, even to undo it. And read the shape of a
diff before reading its content: a count difference is almost always the harness, a value difference
is almost always the change.

### 12.6 A sweep that produced no files is not a sweep that found nothing

Master renamed the `W8` qtype to `Q8`. The sweep script still passed `--qtype W8`, so the benchmark
printed its usage text and exited zero on every single invocation. The script completed, the task
reported success, and the analysis over an empty directory printed a clean empty table.

Any script that collects measurements must fail loudly when it collected none; one line does it. This
is the same family as 11.15 and 11.16: the dangerous outcome is not an error, it is a well-formed
report of nothing.

### 12.7 Sample the range densely before choosing a threshold

The Rows2 window was first re-measured at every other width, which showed gains at T=6..14 and zeros
at 16 and 17 - and led to a conclusion that the window should be narrowed to 14. Measuring every
integer reversed it: 16 and 17 gain 1.5-1.6 % while 12, 14 and 15 read zero.

The zeros are the instrument. Every absolute median in that sweep is an exact multiple of 2.048 us,
and at T=12 one step is 2.08 % of the operation, so a cell reading zero means both arms landed on the
same step. A sparse sweep across a quantized instrument does not measure a curve; it samples a
staircase, and which cells look flat depends on where the samples happen to fall.

Check the quantum first - it is visible in the absolute numbers, never in the ratios - and then
sample every point in the interval the threshold will move.

### 12.8 No AI attribution anywhere in an upstream submission

Asked on both #250 and #253, in the maintainer's own words: he does not want AI tools listed as
authors or co-authors of commits, and the human author attribution is to be preserved.

So, for anything that goes to his repository:

* **no `Co-Authored-By:` trailer** in any commit message, and it must be absent from the final
  squash message too - a squash merge assembles its message from the branch commits, so a trailer
  left on any one of them reappears there;
* **no "Generated with ..." line** in the pull request body either. He did not ask for that one to
  go, but it is the same signal in a different place, and the point is to stop raising the subject
  rather than to litigate where the line may stand.

This is not a disclosure rule and it is not a judgement on using the tool. His CONTRIBUTING says
plainly that "tool choice neither qualifies nor disqualifies a contribution"; what it demands is
that the contributor understand the whole diff, explain the design and its boundary conditions, and
answer review. The objection is narrower than it looks: `Co-Authored-By` is not a note, it is git
metadata that makes a second author of record, visible in the history forever.

Strip both before pushing, not after being asked.

### 12.9 Compare against what the dispatcher will choose after the change lands

The rejected revision of the ragged-tail PR sent ragged SwiGLU widths to the fused kernel, where the
public `linear` + `silu_mul` composition was 8.5 to 11.6 % faster. The reason it was missed is worth
keeping, because every gate we run was clean.

The PR itself made the alternative faster. The composition's own `linear` gained the ragged TMA path
from this very change, and the routing decision was then made against the baseline *as it had been*.
The maintainer's control row showed it plainly: at the width just below the new floor both arms read
the same number, because both were already taking the improved composition.

So the comparison for a dispatch change is not "new route against master". It is **every
implementation the dispatcher could select at that width, each built from the tree the change
produces** - including the ones the change itself speeds up. Run it through the public entry point,
so what is compared is what production would run.

Two corollaries that cost us a second round here:

**An arm must implement the condition being shipped.** One suite was collected with the *rejected*
floor still in it, which left the band the new condition newly claimed completely unmeasured - and
extrapolation put that band at a 13 to 18 % regression, worse than the defect already rejected.
Check what each arm's binary actually does before trusting a table drawn from it.

**If the right condition depends on another pending change, do not ship the condition.** Offering the
maintainer both tables and letting merge order decide sounds even-handed; it is not, when one of the
two orders ships a regression. Split it: land the mechanism with the dispatch untouched, which is
correct under every order, and reclaim the routing afterwards against the tree that actually exists.

### 12.10 Replacing a precondition with a convention is removing it

The same PR deleted the whole-tile check in the quantizer - correctly, since padding made the old
condition obsolete - and put nothing in its place. The new contract was "the caller allocated through
`allocate_nvfp4_w4a4_workspace`", which nothing enforced: the workspace is two raw pointers, so a
caller sizing its own would have had 255 tokens of scales written past the end, silently.

When a change makes a guard obsolete, the question is never whether to delete it. It is what the new
invariant is and where it is now checked. Here the plane's extent had to travel with the pointers so
the check could live where the old one did.

### 12.11 `git checkout --` restores from HEAD, not from before the damage

A strength control damages a file, builds, runs, and puts the file back. Ours put it back with
`git checkout -- <file>`, which restores the committed state - and the change being tested was
uncommitted, so the restore deleted it. The next build failed with a type error between two files
that had been consistent five minutes earlier, and the only reason the work was not lost is that the
patch script that produced it was still on disk.

Save by copy and restore by copy, compare the two with `cmp`, and put the restore in a `trap` so it
also runs when the script exits early:

```sh
SAVE=$OUT/file.orig
cp "$FILE" "$SAVE"
restore() { cp "$SAVE" "$FILE"; cmp -s "$SAVE" "$FILE" || { echo "RESTORE MISMATCH"; exit 1; }; }
trap restore EXIT
```

The same applies to the other direction - copying one arm's test sources into the other's tree to
compare identical widths. 12.5 already says to rebuild the restored arm; this says not to restore it
with git in the first place.

### 12.12 A strength control has to be louder than the criterion, not louder than the arithmetic

The first control scaled the fused SwiGLU epilogue by 1.001 and **every case passed**, which reads
exactly like a gate that cannot see the kernel at all. It was not: the A4 profile's relative-L2
allowance is `1.6e-1`, because the weights are four bit, so a per-mille perturbation sits three
orders inside the criterion.

Size the damage against the tolerance the test actually applies, and read that tolerance out of the
source before choosing the number. A control that fails to fail teaches nothing, and worse, it takes
a while to tell apart from a gate that is genuinely blind.

### 12.13 `ctest --output-on-failure` prints nothing when everything passes

A witness that scrapes test stdout for statistics collects zero records from a green suite, because
that flag shows output only for failures. The emptiness guard from 11.16 caught it, which is the
only reason it is a footnote rather than a retraction. Use `ctest -V` for anything that reads test
output, and strip the `N: ` prefix it puts on every line - the number differs between arms, so a
naive comparison reports every record as changed.

### 12.14 An outlier gets a rerun before it gets an explanation

Three widths in the band this change deliberately leaves alone read +0.57 to +0.85 % in one sweep,
against nulls of 0.15 to 0.32. Both arms take the same route there, so there was nothing in the
dispatch to explain it, and a plausible mechanism was to hand: the benchmark sizes one arena from
the whole sweep and hands it to every width, and the change cuts that arena by an order of
magnitude, so different addresses and different cache behaviour at widths the dispatch never
reaches.

That story was written up before it was tested. A narrow sweep over the band alone was consistent
with it. Then a third session - four arms, same extent for all of them - read the same three widths
at 0.00, 0.00 and -0.14 %, worst cell in the band +0.15 %. **The outliers were not a property of
the change and not a property of the arena. They did not reproduce.**

Two things to carry. An outlier in a single session is a candidate for a rerun, not a candidate for
an explanation; the explanation is what you write once it survives one. And a mechanism that merely
fits the observation is not established by an experiment that fails to contradict it - 11.4 already
says this about sentences in a body, and it applies just as much to a paragraph written for our own
notes.

### 12.15 Pick a threshold where the sign is stable, not where the crossover is

The fused SwiGLU floor was measured four ways. Every session put the sign flip at the same place -
T=263 slower, T=264 faster, a tail of eight tokens - which is as clean a crossover as this bench
produces. And at T=264 itself four readings give **-0.27, +0.59, +0.31 and -0.08 %**: the crossover
is exactly where the two implementations cannot be told apart, so it is exactly where a threshold
is least reproducible.

What is stable is the ground either side of it. Below a tail of eight, every session agrees the
composition wins, by up to 0.69 %. From a tail of sixteen upward every session agrees the fused
route wins, by 0.5 % rising to 6.8 % at the top of the band. A threshold belongs in the stable
region, and the distance from the crossover is the margin you are buying against the next machine.

6.7 already says a tuned constant must be justified on the binding case or kept conservative. This
is the operational form of it: measure the crossover to know where it is, then do not put the
constant there.

### 12.16 Report the alternative you rejected, with its numbers

The choice here was a floor at one M tile - worth a further median of 0.77 % over 63 widths, at the
cost of seven widths regressing up to 0.69 % - against a floor at two, which reads +0.03 % median
across that band and regresses nothing. Both were measured in one session, four arms, so the
comparison is paired rather than assembled from two runs.

Shipping the conservative one and saying nothing would have invited exactly the review that killed
the first version of this work: *"you left a gain on the table"* is the mirror of *"you shipped a
regression"*, and both are answered by the same table. Put it in the body, name what the other
option costs, and say which machine the crossover was measured on.

This is not the same as offering the maintainer a choice between two outcomes, one of which is bad
- 12.9 rules that out. Both options here are safe; one is more conservative, and the evidence for
preferring it is stated rather than assumed.

### 12.17 One figure, three populations: say which one you are quoting

At T=256 this package can report −21.70 %, −21.97 % or −21.83 %. All three are honest. The first is
the dense sweep, the second the wide sweep, the third pools both — and the pooled one is what the
body's own summary tables are computed from, because the 38-of-38 count is over the pooled band.

I corrected a narrative sentence to the dense figure while leaving the tables on the pooled one, and
for two rounds the body quietly disagreed with itself two screens apart. Nothing was wrong; nothing
traced either, and a reviewer checking a headline number against the table below it would have found
a mismatch I could not explain on the spot.

The rule is not "pick the pooled one". It is: a body quotes one population throughout, names it once
("those figures pool the two sweeps"), and shows the other readings beside it so the spread is
visible rather than hidden. Three sweeps agreeing to within 0.27 % is evidence. Three sweeps quoted
interchangeably is a defect.

### 12.18 Evidence borrowed from another submission has to travel with a label

This body cites an identical-baseline decode arm at +0.55 % median. That arm was never collected for
this change — it belongs to a different submission's package, on the same machine and the same build
configuration. The number is the right one to quote: it is a property of the stand, and the stand is
what decides whether an end-to-end claim is possible.

What was wrong is that nothing in `raw/` carried it. The audit caught it as three percentages with no
file behind them, which is exactly the shape of a number nobody measured — even though somebody had.

If a figure comes from another package, copy the raw into this one, say in the body that it was
collected elsewhere and why it still applies, and state what it is evidence *of*: the instrument, not
the change. Otherwise the strongest argument in the submission — *"I am not claiming an end-to-end
number, and here is the measurement that says I cannot"* — rests on a figure the reviewer cannot
check.

And when you go to fetch it, check whether there were two collections. There were: an earlier one at
two repetitions read +1.81 % median with a +5.41 % worst pass. 12.16 says to report the alternative
you rejected. A rejected *instrument* counts.

### 12.19 Structure is looked up, not recalled

I wrote "both registered text geometries are in play" from memory. The tree has five model cards over
three base models, and I had artifacts for two of them. The sentence was not false, but it was an
assertion about the shape of the world made without opening anything.

Fetching the three published configs took one command and turned a hedge into the strongest claim in
the submission: all three are head_dim 256 with a 64-channel rotation, and the two head geometries in
the predicate are the only two that exist. The same lookup produced the fact the package was missing
entirely — the call site is on the full-attention path, so it fires 16 times per pass of 64 layers,
or 10 of 40, not once per layer. Every per-pass number in the body depends on that multiplier, and I
had been about to ship without it.

The general form: any sentence of the shape "there are N of X" or "X always has property P" is a
lookup you have not done yet. Do it before the reviewer does, because they have the tree open.

### 12.20 A live null does not make a number portable between machines

The PV block-buffered package is the cleanest counterexample we have to our own habit of treating
"the null arm was green" as a warrant for the figure. Both campaigns had a working null, and the
figure still moved by a factor of two.

The measurement carries **two independent nulls inside one run**, not one:

* `nvfp4` and `k8v4` are control formats — the route does not fold on them, so the change is
  required to read exactly zero. It read **0.00 %** on both, in both passes.
* Both passes agreed to the third digit.

So the instrument was demonstrably sound. Then:

| | base `ad0f3d38`, rented stand | base `b88c0f6`, our stand |
|---|---|---|
| headline | ×1.20…1.29, i.e. **−17…−22 %** | |
| bf16 | | **−10.68 %** |
| int8 | | **−16.82 %** |
| fp8 | | **−16.67 %** |
| nvfp4 (control) | | **0.00 %** |
| k8v4 (control) | | **0.00 %** |

Not one format landed inside the old band. Two landed just below its lower edge and the third at
roughly half of it. Had the old headline gone into the body — the package was assembled to submit
it — the maintainer would have read ×1.29 and measured ×1.12.

**The rule.** A null arm proves the two sides of *this* run differ by more than the instrument. It
says nothing about whether the figure survives a change of machine, driver, clock policy or base
commit. Those are not noise; they are different populations. So:

* A figure is quoted with the machine and the base commit it was taken on, every time.
* A package that has sat while master moved is **re-measured**, not re-read. Its old numbers are
  evidence that the direction is real, not that the magnitude is.
* When an old and a new campaign disagree, the body carries the new one and says the old one
  existed and what it read. 12.16 already requires reporting the alternative you rejected; a
  superseded measurement of your own work is one.
* Do not average across campaigns or quote the friendlier of the two.

This sharpens 12.7 (one stand per report) rather than repeating it: 12.7 says do not mix stands
inside one table, and this says a green null inside one stand is not a licence to carry the number
out of it.

The raw numbers above were measured by session `llm-5090-3b`; they live in
`pr_pv_block_buffered/` (`README.md` for the rented-stand campaign, `ГОТОВНОСТЬ_2026-09-10.md` for
the remeasure, `raw/attn_{base,pv}_p{0..7}.txt` with a `.witness.txt` per pass).

A third measurement of the same cells is in flight, on `5b4303c0`. Note what it does and does not
isolate: it runs on the **same card, same driver 616.64, same 575 W cap** as the second campaign, so
it varies the base and the branch, not the machine. If it lands on a third value with the control
formats still reading zero, the machine is ruled out and the base commit is left holding the
difference - which is the stronger finding of the two, because a base commit is something we choose
and a rented machine is not.

### 12.21 Portability is a property of the quantity, not of the measurement

12.20 says a green null does not make a figure portable. That is true but too broad, and the
correction is measured rather than argued. Three kinds of number behave differently across a change
of machine or base:

**Accuracy figures transfer exactly.** The PV package's cell `causal batch d256-h24-kv4 bf16 W=16
B=1 phase=0` read 0.5977 → 0.6107 on a rented stand on 2026-09-10 and the same 0.5977 → 0.6107 on
our card today. They are deterministic; there is nothing for a machine to change. Measured by
`llm-5090-3b`.

**End-to-end figures transfer closely — but read what the number is before citing it.** The GDN
change read **+0.213 %** decode [+0.123 … +0.277], 6 of 6 passes, on our stand, against **+0.228 %**,
6 of 6, on rented stand B at `b88c0f6`. Both at `-r 4 --warmup 2`, six passes, mirrored arm order.
Raw in `/root/e2e_gdn`; measured by `llm-5090-3b`.

I first wrote that pair up as evidence that end-to-end figures transfer. It is weaker than that, and
the arithmetic says so. The null arm read +0.054 % [−0.071 … **+0.151**], and the effect's band
starts at +0.123 — **the two overlap over [+0.123, +0.151]**. The effect does not separate from its
own null; it holds on the sign criterion alone, 6 of 6, p = 1/64. So the two campaigns agreeing to
0.015 pp is agreement an order of magnitude finer than the instrument's own null is wide
(0.222 pp). With two campaigns you cannot tell that apart from luck.

The companion figure behaves better and shows what a citable one looks like: PV prefill on 35B read
+0.999 % [+0.852 … +1.522], 6 of 6, against a null of +0.099 % [−0.401 … +0.721] — separated, but
by **0.131 pp**, which is the number to quote alongside it rather than the bare +0.999 %.

So the discipline: before citing an end-to-end figure as transferable, check whether its band clears
its null's band, and say which of the two cases it is. "Reproduced in magnitude and on the sign
criterion" and "separated from the null" are different claims, and only the second licenses the
word *effect*.

**Operator figures do not.** GDN's operator read −29.7 % at T=16 and −26.98 % today. PV's bf16 cell
went −22 → −10.7 → −12.6 % across three campaigns. This is the case 12.20 was written from.

The mechanism is plausible and matches the shape: an end-to-end figure is a ratio of two whole
passes, in which the change is a small part and most of the common time cancels. An operator figure
is a ratio of two short kernels, where everything the machine does around them is the measurement.

**But an end-to-end figure is only portable once it is sampled enough to be portable, and that is
not free.** From this package's own record: the identical-baseline decode arm was collected twice on
one machine, one base, one build. At `-r 2 --warmup 1` it read **+1.81 % median with a +5.41 % worst
pass**. At `-r 10 --warmup 3` the same arm read **+0.55 % median over −0.44 … +0.89**. Nothing
changed but the sampling. So "end-to-end transfers" is a statement about adequately sampled
end-to-end figures; an under-sampled one is not portable even to itself, and the way to find out is
the null arm, not the effect.

So:

* Quote an accuracy figure freely; say which build produced it and stop there.
* Quote an end-to-end figure across machines only with its pass count, its warmup, and its null
  arm's spread beside it. Without those three it is not evidence that it transfers.
* Never carry an operator figure across a machine or a base. Re-measure.
* When the three disagree about whether a change helps, the operator figure is the one that moved,
  and it is the one to re-take.

One more distinction this exchange produced, about branches rather than numbers. Two of our open
changes edit `src/models/qwen3_5/execution/text.cpp`. `git cherry-pick -n` of both onto master is
clean, which says the *text* does not collide. It does not say the *meaning* does not: the hunks sit
in the same two functions, `mtp_forward_tail` and `attn_mix`, about twenty lines apart, and a symbol
one body cited by line number moves from `text.cpp:1080` to `:1088` depending on which lands first.
A clean cherry-pick and a shared file are two facts, and the first does not retire the second. Cite
functions and symbols, not line numbers, whenever another open change touches the same file.

### 12.22 Agreement of sign between passes is not agreement

A scan over dispatch tables looked for widths where the table's own neighbour is faster. The filter
was: the drop must appear in both passes with the same sign. It produced 62 candidates, headed by
`35b_mtp_proj` at T=10, **−49.16 %**.

That cell's two passes read **14.336 µs** and **39.168 µs**. They agree on the sign and disagree by
a factor of 2.7. The headline finding was noise that happened to fall the same way twice.

Replacing the filter with two conditions — the passes of a cell must agree **with each other**, and
the drop must exceed the **worst spread of the two cells being compared** — left 20 of the 62. Every
large "finding" was in the 42 that went. What survived is smaller and real: `27b.mtp_attention` q8
`[14336,5120]` at 56→57 reads 120.679 → 74.469 µs, **−38.29 %**, with a pass spread of 0.65 %.

Measured by `llm-5090-3b`.

**The rule.** A sign test answers "did it move the same way twice", which is the wrong question when
a single pass can be wrong by a factor of two. Before a delta between two cells is a candidate:

* each cell's passes agree with each other, and you state by how much;
* the delta is larger than the worst of the two cells' spreads, not larger than the average;
* the spread is printed next to the delta, so a reader can apply the test rather than trust it.

This is the same discipline the null arm enforces for A/B work (12.20, 12.21), applied where there
is no null arm because both sides are real configurations. The pass spread *is* the null there, and
it has to be printed for the same reason.

It also explains a shape of error we keep repeating: the biggest number in a scan is the most likely
to be junk, because a scan ranks by magnitude and noise has the largest magnitude. When a scan's
headline is also its least reproducible cell, that is the expected outcome, not a surprise. Sort the
survivors by delta-over-spread, not by delta.

**And check whether the source already admits the finding.** The strongest of these did not need a
measurement to be believable — `select_q8_n34816_k5120` in
`src/ops/linear/q8/shapes/n34816_k5120.cu` reads:

```
    if (tokens <= 48) return launch_q8_mma_r64x16_c48_k128_a1;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_mma_r128_c64;
```

MMA, then back to k-split, then MMA again. A table that reverses itself over eight tokens is a claim
about the machine that some measurement once made and nobody has re-made. Reading it costs nothing
and tells you where to point the instrument.

#### 12.22a A tool that reports absence has to be shown a case where presence is certain

Two failures in one evening, different tools, identical shape.

**The scan filtered on sign.** Two passes of a cell agreeing in direction was treated as the cell
being real. It reported 62 candidates. The filter had never been shown a cell that was known to be
noise, so nobody knew it would pass 14.336 µs against 39.168 µs.

**The grep reported no gate.** Reading `uses_a8` across five shape files, a one-line `sed` pattern
was used against what turned out to be a two-line function body, and `n34816_k5120` printed as
`<none>`. The conclusion drawn and passed on was "this shape has no gate at all". It has
`min_tokens == 1 || max_tokens >= 5`, so A8 does not run at T = 2, 3, 4 — visible in the data as
a16 and a8 columns agreeing to three digits at T=2 and T=4 and diverging at T=1.

In both cases the instrument answered "there is nothing here" and was believed, because an empty
answer looks like a clean answer. A false negative has no symptom: a wrong number invites a second
look, a missing row does not.

**So: before trusting a tool that reports absence, run it against a case where presence is certain.**

* A grep or a parser: point it at a line you have read with your own eyes and confirm it comes back.
  If five files are being scanned and one prints empty, that is the file to open, not the file to
  skip.
* A filter over candidates: feed it a cell you know is noise and confirm it is rejected. A filter
  never tested against a known negative is a filter with an unknown false-positive rate, which is
  the same thing as no filter.
* A collection step: assert a non-zero count before analysing. `NINFER_OP_REPORT_STATS` with
  `ctest --output-on-failure` collects nothing from a green suite (12.13) — the emptiness guard is
  what caught that, and it is the same guard as this rule.

The cheap version of this costs one command. The expensive version is telling a colleague, or a
maintainer, that something is not there.

**The typo case deserves its own line, because it is the most common one.** Searching for
`mma_nvf4` where the symbol is `mma_nvfp4` returns nothing, and nothing is exactly what a genuine
absence returns. The conclusion drawn was "the wrapper exists but is dead"; the wrapper has three
callers. One dropped letter, and a whole backlog item was about to be scored on the wrong premise.

A probe catches this and nothing else does: a misspelled pattern fails against the known-present
case too, so the probe comes back empty and the search is condemned before its verdict is used. A
spell-check of your own pattern is not available; a probe is.

Three false negatives in one night on one project, all three from the searcher's own tools, none
from anyone's data. That ratio is the point. We check data we are given; we do not check the
instrument we reach for, because reaching for it feels like looking.

Two of the three had consequences worth naming. `fp8_a16_ksplit_mma` was reported as included by
exactly one shape; it is included by three (`shapes/n248320_k5120.cu`,
`gdn_input_proj/fp8/fp8_gdn_input_matrix.cu`, `attn_input_proj/fp8/fp8_attn_input_a16_small_t.cu`).
And "prefix reuse is absent upstream" was wrong about a subsystem of seventeen files with
`ExecutionOptions::allow_prefix_reuse` defaulting to true and six `PrefixReusePath` kinds; what is
absent is *position-independent* reuse, which is a claim a tenth the size. Found by `llm-5090-3b`.
