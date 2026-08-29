# PR #113: ответ на замечание Codex-бота

Состояние на 29 августа, 12:10. **#112 — ревью бота прошло без замечаний.** **#113 — одно
замечание, P2.** Мейнтейнер ни там, ни там пока не отвечал.

## Что сказал бот (thread `discussion_r3886491016`, строки +39..+40)

> **Add coverage for a newly fused token width.** This predicate changes both the workspace path and
> floating-point arithmetic for 256, 512, 768, and larger multiples, but
> `tests/ops/linear_swiglu/test_nvfp4.cpp` still tests only `{5, 48, 49, 128, 1024}`; 1024 already
> used the TMA route before this commit. Consequently, the reported `ctest` run exercises none of
> the newly routed cases and cannot catch launch, workspace, or numerical regressions in this
> change. Adding at least T=256 would cover the new predicate without increasing the test oracle's
> existing maximum extent of 1024.

**Замечание верное, и это ровно тот вопрос, который мы сами собирались задать** (в наших черновиках
он был «вопрос 2»: входит ли расширение `kA4Cases` в заявку). В отправленном теле PR вопрос не
остался — бот задал его за нас. Спорить не с чем: `ctest` на ветке действительно не исполняет ни
одной новой ширины.

## Правка

Один символ списка, `tests/ops/linear_swiglu/test_nvfp4.cpp:13`:

```diff
-        constexpr std::array<std::int32_t, 5> kA4Cases{5, 48, 49, 128, 1024};
+        constexpr std::array<std::int32_t, 6> kA4Cases{5, 48, 49, 128, 256, 1024};
```

Почему именно 256 и только 256:

* это самая узкая ширина, которую впускает новый предикат, поэтому она исполняет **обе** новые
  ветки сразу — и новую ногу `resolve_route`, и новую ногу
  `nvfp4_linear_swiglu_workspace_capacity_bytes`;
* это `T`, кратное блоку, но **ниже** старого `kPrimaryT`, то есть случай, которого старый код не
  мог породить в принципе;
* критерий оракула не трогается, максимальная ширина списка остаётся 1024 — ровно то, о чём просил
  бот;
* обе руки её проходят, поэтому тест осмысленный, а не «зелёный только на ветке».

## Текст ответа в тред (после проверки — заполнить `TBD`)

> Fair, and it is the right objection: `ctest` on this branch does not execute the predicate the
> branch adds. Pushed as TBD_SHA — `T = 256` joins `kA4Cases`.
>
> 256 rather than a wider multiple because it is the narrowest width the new predicate admits, so it
> is the one case that exercises both new legs at once: the new leg of `resolve_route` and the new
> leg of `nvfp4_linear_swiglu_workspace_capacity_bytes`. It is also a `T` below the old `kPrimaryT`,
> which is a shape the old route could not produce at all. The criterion is untouched and the case
> list's maximum extent stays 1024.
>
> It is also not decoration. Moving the route without widening the capacity query makes the shipped
> Op test report `exact workspace query/execution high-water mismatch` at 256, 512 and 768 - that is
> how I found the second half of this change. `T = 256` is the case that would have caught it, and
> without it nothing in `ctest` would.
>
> It is a case both arms pass, which is what makes it a regression test rather than a branch-only
> one. With `NINFER_OP_REPORT_STATS=1` at `T = 256`, `master` reports relative L2 0.059000 against
> the FP64 oracle — 36.9% of the criterion — through `LinearW4A4Post`, and this branch reports
> 0.058382, 36.5%, through the fused route; the gross ratio moves 0.531 -> 0.486. Both print `OK`.
>
> `cd build && ctest -j1`: TBD_CTEST_BRANCH on the branch with the case added, TBD_CTEST_MASTER on
> `master` built in the same directory in the same run with the same case added.
>
> The submission is now two files: the route table and this one line of the test.

## Что нужно проверить до отправки

1. Собрать **голову PR** (`fe6f70d1` на `ce09aee5`), а не нашу локальную `436aba7b` — числа должны
   сниматься там, куда бот смотрит.
2. `ninfer_linear_swiglu_nvfp4_test` с добавленным 256 — на ветке и на `master` с той же правкой
   теста. Обе должны печатать `OK`; забрать `rel_l2` из `NINFER_OP_REPORT_STATS=1`.
3. Полный `ctest -j1` на обеих руках в одном прогоне.
4. Сверить, что цифры 0.059000 / 0.058382 / 0.531 / 0.486 воспроизводятся — они сняты на `1fc1cb76`,
   а голова PR стоит на `ce09aee5`.

Пока это не сделано — в тред ничего не писать.

## Важно: `branch.bundle` в этой папке отстал от головы PR

Голова PR — `fe6f70d1` на `ce09aee5`. Сверил её диff с нашей локальной веткой `436aba7b`: они
**различаются**, и различие содержательное, а не косметическое.

| | наша ветка `436aba7b` | голова PR `fe6f70d1` |
|---|---|---|
| граница маршрута | `tokens >= kPrimaryT` (1024) | `tokens >= kTmaBlockM` (**256**) |
| запрос ёмкости | `max_tokens >= kPrimaryT` | `max_tokens >= kTmaBlockM` |
| `kPrimaryT` | оставлена | **удалена** |

То есть подана **опущенная граница** — ровно то, что в наших черновиках стояло вопросом 3 («не
должна ли граница быть `kTmaBlockM`»). Напарник ответил на него сам, опустив её и приведя в теле PR
измерения на 256/512/768 (×1.117 / ×1.366 / ×1.293). Наши замеры опущенной границы это
подтверждают.

**Следствие: все проверки снимать на `fe6f70d1`, а не на нашей ветке.** `branch.bundle` здесь —
исторический артефакт первого варианта, обновлять его не нужно, но и опираться на него нельзя.
