# Черновик комментария в issue #27

Не PR. Комментарий. Правка не наша — автор `sergiuszm`, реализация `sergiuszm/ninfer-5090@4483c820`,
авторство коммита при любом использовании остаётся его.

Заполнять `TBD_` только из `/root/qual/gPV`. Пока в тексте есть хоть один `TBD_` — не отправлять.

---

> Independent reproduction, on a second RTX 5090 and a different model. To be clear about what this
> is: the change is yours, I did not write any of it - I cherry-picked `4483c820` onto current
> `master` and measured it. What I can add is a machine you do not control, a model your numbers do
> not cover, and an answer to the arch-guard question that is a measurement rather than an opinion.
>
> RTX 5090, sm_120a, driver 580.105.08, CUDA 13.1.115, Release, `-DCMAKE_CUDA_ARCHITECTURES=120a`,
> base `master` `1fc1cb76`. The cherry-pick is still clean there: two files, +34, no conflicts. The
> two `master` commits that have touched those files since (`6183c9be` fp8 KV cache, `17a7275f`
> int8 KV Hadamard rotation) do not reach the lines the change edits.
>
> ## The arch guard, since you asked
>
> You offered three shapes and said you would follow a preference. On a build that compiles only
> `sm_120a` the choice is free, and that is checkable rather than arguable: I built all three -
> your guard as written, `== 1200`, and no guard at all - and compared the SASS body by body.
>
> TBD_GUARD
>
> So option 1 costs nothing on this product and is the one that leaves no dead `#else` behind. The
> Ada leg is the only thing the wider guard buys, and whether that matters is a question about the
> project rather than about this kernel.
>
> ## Qualification, with the headroom rather than only the verdict
>
> `ninfer_softmax_attention_test` passes on both arms, as you reported. `NINFER_OP_REPORT_STATS=1`
> also prints how much of the criterion each case actually uses, which seemed worth recording since
> "passes" and "passes with 3% of the limit left" are different claims:
>
> TBD_ORACLE
>
> `ctest -j1`: TBD_CTEST.
>
> ## Operator
>
> `ninfer_causal_softmax_attention_bench`, `--entry append`, 2,048 new tokens, INT8 KV. **The bf16-KV
> rows are the control** - `prompt_i8.cuh` is the INT8 path, so bf16 has to stand still, and it is
> the one arm in this table that the change cannot reach.
>
> TBD_OPERATOR
>
> ## End to end, on a model your numbers do not cover
>
> Qwen3.6-35B-A3B, INT8 KV, 4,096-token prefill chunk, greedy, arms alternating inside each round,
> every point its own process. I quote the paired per-round ratios rather than a mean of means
> because the absolute rate drifts between rounds on both arms and the ratios do not.
>
> TBD_E2E
>
> Our configuration is not yours and the two columns should not be averaged together: you measured
> operator TFLOP/s and end-to-end throughput on your own build (185.9 -> 226.3 at 8K, 191.9 -> 236.0
> at 64K, 192.5 -> 229.6 at 128K; prefill 2,247 -> 2,299 at 64K, 1,896 -> 1,964 at 128K,
> 1,607 -> 1,682 at 200K). This is a different model, a different chunk width and a different
> harness. Both columns say the same thing about direction.
>
> ## Generation-level accuracy, and what it can and cannot tell you
>
> One thing the issue does not have is a generation benchmark, so here is ours, run through EvalScope
> on the full sets: AIME25 (30 problems) and GPQA-Diamond (198 questions), concurrency 2, identical
> settings across configurations.
>
> | configuration | AIME25 | GPQA-Diamond |
> |---|---:|---:|
> | A - stock, bf16 KV, chunk 1024 | 90.0% | 84.34% |
> | B - summation order only (chunk 8192, wider MoE tiles) | 90.0% | 84.34% |
> | C - B plus INT8 KV plus this change | **90.0%** | **84.85%** |
>
> The row that matters is not C, it is A against B. Those two differ **only** in the order of
> summation, and they land on the same score on both sets - 167 of 198 in both GPQA runs. That is
> the measured noise floor of the harness, and it is what makes C an experiment rather than an
> anecdote.
>
> **The limitation, stated before anyone has to ask:** 198 questions puts one sigma at about 2.5%,
> and one AIME problem is 3.3%. This bench rejects a degradation larger than roughly 5%; it cannot
> confirm the absence of a 1-2% one. That is a bound set by the number of questions, not by how the
> runs were done. TBD_ACC_RERUN
>
> ## One question back, since it is your idea
>
> The same ceiling looks like it applies to `prompt_bf16.cuh`. On bf16 KV that kernel is 41.6% of
> prefill on the trace I took today - by a wide margin the largest single kernel - and its PV stage
> is `mma_bf16` into a `float acc[PVNt][4]`, which is the same f32-accumulate half-rate path your
> issue is about. It is harder there than in the INT8 route: `prompt_i8.cuh` already has `v_f16` in
> shared because it has to dequantise, while the bf16 route `cp_async`s V straight from the cache,
> so the f16 operands would have to come from a register-side conversion after `ldmatrix` (which is
> exact in that direction - bf16 carries 7 mantissa bits and f16 carries 10 - but narrows the
> range).
>
> Is that something you are already planning? I am not going to start on it: it is your mechanism
> and the first move on it should be yours. If you would rather someone else measured it, say so and
> I will, with the attribution where it belongs.
>
> ## What I did not check
>
> TBD_LIMITS
>
> Happy to re-run any of this differently. If a PR is wanted, it should be yours - it is your change,
> and the commit author should stay the way it is.

---

## Что нужно снять до отправки (кампания `/root/pv_arms.sh`)

1. `TBD_GUARD` — census SASS: `pv` против `pv1200` и против `pvnone`. Ожидание — «identical device
   code» в обоих сравнениях. Если вдруг не так — писать как есть, это и будет ответом.
2. `TBD_ORACLE` — строки `OP_ERROR_STATS` для внимания на обеих руках, колонка `rel_l2_ratio`.
3. `TBD_CTEST` — полный прогон на ветке.
4. `TBD_OPERATOR` — таблица TFLOP/s или мкс, int8 против bf16, 8K/32K/64K/128K, обе геометрии.
5. `TBD_E2E` — 3 раунда, чередование, int8 и bf16, 16K/32K/64K. **bf16 обязан стоять на 1.000 и
   давать побитово тот же вывод.**
6. `TBD_ACC_RERUN` — перепрогон стенда на текущем master либо честная фраза, что таблица снята
   24 августа на другой базе и не перемерена.
7. `TBD_LIMITS` — чего не делали: нет счётчиков `ncu` (`RmProfilingAdminOnly`), не мерили Ada,
   не мерили fp8-KV, спекуляция выключена.
