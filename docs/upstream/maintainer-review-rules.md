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
request."* Merged PRs in this corpus carry a `Co-Authored-By: Claude Opus 5` trailer and a
"Generated with Claude Code" line. Disclosure alone is not enough — *"Generated code that has not
received these basic checks is not ready for submission."*

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
