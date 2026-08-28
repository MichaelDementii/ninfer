# Правила автора: что он написал дословно и откуда

Три источника, все проверены по живым страницам GitHub 2026-08-28/29.

---

## 1. `CONTRIBUTING.md` на master — правило «сначала Issue»

Появилось после закрытия #96. Это самое важное изменение процесса.

> Every bug, performance opportunity, feature request, protocol change, and architecture proposal
> **must begin with an Issue before implementation starts**. … **Wait for the maintainer to confirm
> the scope and implementation direction** before investing in a pull request. Opening an Issue does
> not by itself approve a proposed design. A pull request without a linked, confirmed Issue **may be
> closed without detailed review**.

### Что обязан содержать перформанс-репорт

> A performance report must identify **the level of the claim: operator, schedule, request phase, or
> end-to-end inference**. Include the baseline and candidate measurements under comparable
> conditions, the exact workload and commands, hardware and toolchain, warmup and repetition method,
> and a useful summary of the results.
>
> Account for relevant tradeoffs such as **workspace, resident memory, transfer cost, numerical
> quality, and effects on other execution paths**. **End-to-end results can establish an end-to-end
> observation; they do not isolate an operator change.** A proposed operator optimization should
> therefore include a direct operator benchmark and appropriate correctness evidence.

### Что обязан содержать PR

> - the linked Issue and the agreed scope;
> - the concrete design and why it fits the current architecture;
> - the affected behavior, ownership boundary, or public contract;
> - the exact verification commands and summarized results;
> - the workload, hardware, toolchain, and methodology for any performance claim; and
> - **every relevant check that was not run and the resulting limitation.**

### Когда он закрывает без построчного ревью

> - was submitted before its problem, scope, or implementation direction was confirmed;
> - falls outside the supported product or expands an unapproved contract;
> - combines independent changes into an unreviewable implementation;
> - contains evident correctness, ownership, lifetime, state, or boundary problems;
> - lacks evidence appropriate to its correctness, numerical, or performance claims;
> - cannot be explained and supported by its contributor; or
> - would require the maintainer to redesign, split, debug, or complete its core implementation.

И отдельно, чтобы не строить иллюзий:

> The maintainer may preserve the report or core idea and **implement it independently within the
> current architecture**.

---

## 2. Закрытие PR #96, 27 августа — условия новой заявки

> A 38-commit, 61-file performance stack is not reviewable or maintainable as one change.
> Individually described commits do not make the combined state independently verifiable.

> Some of the debugging work was valuable. The Sparse-MoE prefill scan race was a real upstream bug
> and has been fixed independently in `fbd0472`. The proposed GDN BF16 carry, however, is **not** an
> upstream correctness bug: that accumulator is private arithmetic, not a required BF16 state
> boundary. Problems introduced by this branch, such as the MTP profiling cap and global GDN
> completion tickets, should also not be counted as existing upstream bugs.

> The default prefill chunk of 1024 is **deliberate**. The reported workspace grows from about
> 120 MiB at 1024 to 482 MiB at 4096 and 963 MiB at 8192. That memory comes directly out of the
> capacity available for KV, cached contexts, and active requests, so the larger-chunk gain is not
> free and does not justify changing the default.

> A new PR should contain **one mechanism, or one inseparable dependency chain**, rebased on current
> master, with **production-shape operator benchmarks**, **numerical validation**, and **resource
> deltas**. Cross-kernel fusion and prefetch work should additionally measure **the relevant
> producer-consumer or CUDA Graph step**. End-to-end measurements should remain **the final
> confirmation, not the sole evidence**.

> Please do not bundle **product-contract changes** into performance work. Expanding MTP from 5 to
> 15 drafts changes the CLI domain, state/workspace bounds, and Graph profiles; it requires a
> separate proposal and workload-level evaluation. **Hidden mutable channels such as thread-local
> setters or module-global device state are also not acceptable ownership.**

Что он сам назвал приемлемыми темами:

> **Attention/norm fusion, direct GDN Q/K/V output, and individual MoE/projection kernel
> optimizations** are reasonable examples of independently reviewable work.

---

## 3. Ревью PR #99, 27 августа — четыре правки

1. > `CausalConvSplitOutput3` currently keeps row counts, partition selection, and leading dimensions
   > as runtime values in order to support arbitrary partitions. … Please use **compile-time
   > specializations for the registered geometries**, with wrapper dispatch based on geometry, **so
   > the production route does not pay for unused generality.**

2. > The new entry sends `T == 1` through the small-T kernel instead of the existing dedicated
   > decode-shaped kernel, but there is **no short-interval performance evidence**. Please measure
   > both registered geometries at `T = 1/2/7/15/16/17/64/65` and **select the route from those
   > results**, including a dedicated split decode route if the data supports it.

3. > The current microbenchmark compares the packed convolution alone with the split entry. The old
   > complete stage also includes the three extraction operations. Please provide a direct,
   > same-condition comparison of `packed convolution + 3 x extraction` versus `split convolution`.
   > This may be **a temporary decision benchmark**; there is no need to retain the legacy stage
   > after the implementation decision is closed.

4. > The model documents **should not record the concrete function name or whether a private
   > intermediate tensor is materialized**; the model mathematics did not change. Please also keep
   > the generic Op parameter names independent of the first caller (`out0/out1/out2` rather than
   > `out_q/out_k/out_v`) and **enforce the documented state rule** that input/output state storage
   > must be either disjoint or an exact alias.

И процедурное:

> Once the requested changes are ready, please **squash the PR into a single commit and rebase it
> onto the latest `master` HEAD** before final review.

---

## 4. Сухой чеклист перед отправкой

- [ ] Issue открыт, ответ получен, объём подтверждён.
- [ ] Один механизм. Если механизмов два — это две заявки.
- [ ] База — сегодняшний `origin/master`, ветка ребейзнута, один коммит после squash.
- [ ] Уровень заявки назван первой строкой.
- [ ] Операторные бенчи **из дерева**, в продуктовых формах, с прогревом и повторами.
- [ ] Внутри тех же бенчей есть **строки, которых правка не касается** — это контроль.
- [ ] Шумовой пол **измерен**, а не объявлен: та же рука против себя, несколько проходов.
- [ ] Маршруты, которые правка задевает, **просвипованы через границы диспетчера**.
- [ ] Численность: побитово или честный разбор, почему нет.
- [ ] Дельта по ресурсам: регистры, shared, spills, воркспейс, узлы графа.
- [ ] Профиль в продукте — доказательство локальности на реальной нагрузке.
- [ ] Сквозняк — последним, помечен как подтверждение, с разбросом по раундам.
- [ ] Раздел «checks not run» написан честно и полно.
- [ ] Ни `getenv`, ни `thread_local`, ни модульно-глобального состояния устройства.
- [ ] Ни одного продуктового изменения: CLI, воркспейс, профили графа, артефакты.
