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
