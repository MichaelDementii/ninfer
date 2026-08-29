# Черновик комментария в issue #27 — числа заполнены

Не PR. Комментарий. Правка не наша — автор `sergiuszm`, реализация `sergiuszm/ninfer-5090@4483c820`,
авторство коммита при любом использовании остаётся его (`Sergiusz Michalik <lsm@sikom.no>`,
проверено: черри-пик его сохраняет).

Всё ниже снято 29 августа, кампания `/root/pv_arms.sh`, данные `/root/qual/gPV`.
**Единственное, что осталось решить перед отправкой** — перепрогонять ли стенд точности на текущем
master или писать честно, что таблица снята 24 августа на другой базе. Сейчас в тексте стоит второй
вариант.

---

> Independent reproduction, on a second RTX 5090 and a different model. To be clear about what this
> is: the change is yours, I did not write any of it — I cherry-picked `4483c820` onto current
> `master` and measured it. What I can add is a machine you do not control, a model your numbers do
> not cover, an answer to the arch-guard question that is a measurement rather than an opinion, and
> the oracle headroom rather than only the verdict.
>
> RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, Release,
> `-DCMAKE_CUDA_ARCHITECTURES=120a`, base `master` `1fc1cb76`. The cherry-pick is still clean there:
> two files, +34, no conflicts. The two `master` commits that have touched those files since
> (`6183c9be` fp8 KV cache, `17a7275f` int8 KV Hadamard rotation) do not reach the lines it edits.
>
> ## The arch guard, since you asked
>
> You offered three shapes and said you would follow a preference. On a build that compiles only
> `sm_120a` the choice is free, and that is checkable rather than arguable. I built all three — your
> guard as written, `== 1200`, and no guard at all with the `#else` leg removed — and compared the
> SASS body by body:
>
> | comparison | functions | comparable | bodies differing | `causal_attention_prompt_i8` bodies |
> |---|--:|--:|--:|--:|
> | guard as written vs `== 1200` | 2937 / 2937 | 2937 | **0** | 6, of which 0 changed |
> | guard as written vs no guard | 2937 / 2937 | 2937 | **0** | 6, of which 0 changed |
>
> Identical device code, both ways. So **option 1 costs nothing on this product** and is the one that
> leaves no dead `#else` behind. The Ada leg is the only thing the wider guard buys, and whether that
> matters is a question about the project rather than about this kernel.
>
> ## Qualification, with the headroom rather than only the verdict
>
> `ninfer_softmax_attention_test` passes on both arms, as you reported. `NINFER_OP_REPORT_STATS=1`
> also prints how much of each criterion is actually used, which seemed worth recording, since
> "passes" and "passes with 5% of the limit left" are different claims.
>
> Of the 103 reported cases, **11 move, and all 11 are `int8-g64` on the prompt and cached-prompt
> routes** — which is exactly the set `prompt_i8.cuh` owns. The other 92 are identical to the digit,
> including every bf16 and fp8 case.
>
> | over the 11 that move | master | branch |
> |---|--:|--:|
> | relative L2, share of its criterion | 52.7% – 53.6% | **53.4% – 54.5%** |
> | gross ratio | 0.488 – 0.586 | **0.531 – 0.641** |
>
> Every one of the 11 grows, none of the criteria is widened, and the worst case still leaves 45% of
> the relative-L2 criterion unused. Worth saying plainly: the residual gets **larger**, not smaller —
> that is what fp16 accumulation costs, and the number is about one percentage point of the
> criterion.
>
> One thing I noticed while doing this and would have wanted told to me: the tightest cases in this
> suite sit at **94.9%** of their criterion — three `causal_softmax_attention_cached d256-h24-kv4
> int8-g64` cases — and they are identical on both arms. They are not among the 11 this change
> touches, so this is a pre-existing tightness rather than anything you introduced.
>
> `cd build && ctest -j1`: `100% tests passed, 0 tests failed out of 94`, 1 skipped
> (`27b_load_plan`, which needs a second 27B artifact this box does not have).
>
> ## Operator
>
> `ninfer_causal_softmax_attention_bench`, `--entry append`, 2,048 new tokens, both registered
> geometries, all three KV dtypes. **The bf16 and fp8 rows are the control** — `prompt_i8.cuh` is the
> INT8 path, so 16 of the 24 cells here are dtypes the change cannot reach.
>
> ```
> ./build/bench/ninfer_causal_softmax_attention_bench --entry append \
>     --geometry d256-h24-kv4 --kv-dtype all --tokens 2048 \
>     --context 8192,32768,65536,131072
> ```
>
> `d256-h24-kv4`, your geometry:
>
> | KV | context | master us | branch us | speedup | master TFLOP/s | branch TFLOP/s |
> |---|--:|--:|--:|--:|--:|--:|
> | **int8** | 8K | 2483.20 | 2064.35 | **1.203** | 186.8 | **224.7** |
> | **int8** | 32K | 8791.04 | 7362.56 | **1.194** | 193.5 | **231.0** |
> | **int8** | 64K | 17196.00 | 14363.40 | **1.197** | 194.8 | **233.2** |
> | **int8** | 128K | 34027.50 | 28450.80 | **1.196** | 195.4 | **233.7** |
> | bf16 | 8K…128K | — | — | 0.9956 – 1.0001 | 182.7 – 188.2 | unchanged |
> | fp8 | 8K…128K | — | — | 0.9990 – 1.0001 | 200.3 – 209.0 | unchanged |
>
> `d256-h16-kv2`, which your numbers do not cover, gains more:
>
> | KV | 8K | 32K | 64K | 128K |
> |---|--:|--:|--:|--:|
> | **int8** | **1.257** | **1.265** | **1.263** | **1.264** |
> | bf16 | 0.9958 | 1.0004 | 1.0000 | 1.0000 |
> | fp8 | 1.0000 | 1.0000 | 0.9971 | 0.9986 |
>
> Your 8K row was 185.9 → 226.3 TFLOP/s; this box reads 186.8 → 224.7 on the same geometry and
> entry. At 128K yours was 192.5 → 229.6 and this box reads 195.4 → 233.7. I would call that
> reproduced.
>
> ## End to end, on a model your numbers do not cover
>
> Qwen3.6-35B-A3B, 4,096-token prefill chunk, greedy, arms alternating inside each round, every point
> its own process, three rounds. **The bf16-KV rows are the built-in control.**
>
> | prompt | KV | per-round branch/master | prefill | decode |
> |--:|---|---|--:|--:|
> | 16,441 | **int8** | 1.0452 1.0481 1.0465 | **+4.66%** | −0.03% |
> | 33,031 | **int8** | 1.0724 1.0746 1.0719 | **+7.29%** | −0.08% |
> | 16,441 | bf16 | 0.9988 0.9996 0.9999 | −0.06% | +0.02% |
> | 33,031 | bf16 | 0.9993 1.0014 0.9992 | −0.00% | −0.03% |
>
> I quote the paired per-round ratios rather than a mean of means because the absolute rate drifts
> between rounds on both arms and the ratios do not.
>
> Our configuration is not yours and the two columns should not be averaged together: you measured a
> different model, a different chunk width and your own harness (prefill 2,247 → 2,299 at 64K,
> 1,896 → 1,964 at 128K, 1,607 → 1,682 at 200K). Both columns say the same thing about direction, and
> the operator table above is where they meet.
>
> **Greedy output.** All 12 comparisons that ran are byte-identical to `master` — both KV dtypes,
> both prompts, three rounds. bf16 has to be, since the change is not on that path; int8 did not have
> to be, and on this corpus it is. That is an observation about this corpus, not a guarantee.
>
> ## Resources
>
> `cuobjdump --dump-resource-usage`, `causal_attention_prompt_i8_kernel`, all instantiations:
> registers 120 → 120 against the explicit `__maxnreg__(120)`, shared 1024 → 1024, local 0 → 0. One
> instantiation loses an 8-byte stack frame, 8 → 0; nothing gains one. So the fp16 accumulator does
> not push anything into local memory.
>
> ## Generation-level accuracy, and what it can and cannot tell you
>
> One thing the issue does not have is a generation benchmark. Ours, through EvalScope on the full
> sets — AIME25 (30 problems) and GPQA-Diamond (198 questions), concurrency 2, identical settings:
>
> | configuration | AIME25 | GPQA-Diamond |
> |---|--:|--:|
> | A — stock, bf16 KV, chunk 1024 | 90.0% | 84.34% |
> | B — summation order only (chunk 8192, wider MoE tiles) | 90.0% | 84.34% |
> | C — B plus INT8 KV plus this change | **90.0%** | **84.85%** |
>
> The row that matters is not C, it is A against B. Those two differ **only** in the order of
> summation, and they land on the same score on both sets — 167 of 198 in both GPQA runs. That is the
> measured noise floor of the harness, and it is what makes C an experiment rather than an anecdote.
>
> **Two limitations, stated before anyone has to ask.** 198 questions puts one sigma at about 2.5%,
> and one AIME problem is 3.3%: this bench rejects a degradation larger than roughly 5%, and cannot
> confirm the absence of a 1–2% one. And this table was taken on 24 August on an earlier base — I
> have not re-run it on `1fc1cb76`. The operator, oracle and end-to-end numbers above all are on
> `1fc1cb76`.
>
> ## One question back, since it is your idea
>
> The same ceiling looks like it applies to `prompt_bf16.cuh`. On bf16 KV that kernel is 41.6% of
> prefill on the trace I took today — by a wide margin the largest single kernel — and its PV stage is
> `mma_bf16` into a `float acc[PVNt][4]`, which is the same f32-accumulate half-rate path your issue
> is about. It is harder there than in the INT8 route: `prompt_i8.cuh` already has `v_f16` in shared
> because it has to dequantise, while the bf16 route `cp_async`s V straight from the cache, so the
> f16 operands would have to come from a register-side conversion after `ldmatrix` — exact in that
> direction, since bf16 carries 7 mantissa bits and f16 carries 10, but narrower in range.
>
> Is that something you are already planning? I am not going to start on it: it is your mechanism and
> the first move on it should be yours. If you would rather someone else measured it, say so and I
> will, with the attribution where it belongs.
>
> ## What I did not check
>
> - No `ncu` counters: `RmProfilingAdminOnly` is set on this host.
> - No Ada. The 890 leg of your guard is untested here; the codegen result above says only that it
>   costs nothing on `sm_120a`.
> - The deepest end-to-end point on this box is 33K. Our 64K fixture prepares past the engine's
>   131072 max context and fails identically on both arms, and KV headroom at 131072 is already zero
>   so `--max-context` cannot go higher.
> - Speculation is off in every run.
> - The accuracy table is on an earlier base, as stated above.
>
> Happy to re-run any of this differently. If a PR is wanted, it should be yours — it is your change,
> and the commit author should stay the way it is.

---

## Что снято и где лежит

| | результат |
|---|---|
| черри-пик на `1fc1cb76` | чисто, 2 файла, +34, автор `Sergiusz Michalik <lsm@sikom.no>` сохранён |
| census трёх форм arch-guard | **identical device code** в обоих сравнениях |
| оракул | 103 случая, двигаются 11, все `int8-g64`; 52.7–53.6% → 53.4–54.5% критерия |
| `ctest` | 94/94, 1 пропуск |
| операторный бенч | int8 ×1.19…×1.27, bf16 и fp8 — 0.9956…1.0004 на 16 клетках |
| сквозняк | int8 +4.66% / +7.29%, bf16 −0.06% / −0.00% |
| вывод | 12 из 12 сравнений побитово совпали, включая int8 |
| ресурсы | REG 120→120, SHARED 1024→1024, LOCAL 0→0, стек 8→0 в одной инстанциации |

Сырьё: `/root/qual/gPV`, лог `/root/pv_arms.log`.
