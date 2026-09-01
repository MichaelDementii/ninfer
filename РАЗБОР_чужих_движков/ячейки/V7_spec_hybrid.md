# V7 — почему спекулятивный декод в vLLM разваливается на гибридной модели

## Что это за механизм

Гипотеза автора отчёта («путь верификации пересчитывает состояние GDN, а не откатывает
его») **неверна**. vLLM откатывает состояние GDN снапшотами: кеш состояний выделяется с
`1 + num_spec` слотами на запрос, ядро верификации пишет по одному снапшоту на каждую
черновую позицию, а следующий раунд читает слот с индексом `num_accepted - 1`. Стоимость
отката — ноль дополнительных байт, и она константна по длине контекста.

Линейная по контексту цена лежит **в десяти слоях полного внимания**, а не в тридцати GDN.
В шаге верификации `max_query_len = k+1 > 1`, и оба доступных на sm_120 бэкенда
(FLASH_ATTN→FA2 по умолчанию, TRITON_ATTN как запасной) при `max_query_len > 1` теряют
разбиение по оси KV: у Triton это записано явным условием, у FA2 — тем, что упакованный
GQA/split-KV путь включается только при `max_seqlen_q == 1`, а `num_splits > 1` в FA2
запрещён кодом. При batch=1 сетка запуска падает с ~32 CTA (по 1/16 KV каждая) до 4–6 CTA,
каждая из которых стримит весь KV целиком. На 170 SM это 2–4 % занятости, и измеренная
цена сходится с этим с точностью до 10 %.

Побочно: черновая голова у vLLM для этой модели строится на **полный** словарь
(`config.vocab_size`), урезанная голова 131072 не используется — механизм `draft_vocab_size`/
`d2t` в vLLM существует, но только для EAGLE-3 и DSpark, не для Qwen3.5/3.6 MTP.

## Доказательства из кода

### Откат состояния GDN — снапшоты, не пересчёт

- `vllm/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:105-116` —
  в ядре верификации: `if IS_SPEC_DECODING: i_t = tl.load(num_accepted_tokens + i_n) - 1`,
  затем `state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)`. Начальное
  состояние берётся из слота, соответствующего числу принятых токенов прошлого раунда. Это и
  есть откат: индекс слота, ноль работы.
- `.../fused_sigmoid_gating.py:157-166` — внутри цикла по токенам, на каждом шаге `i_t`:
  `final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)` и
  `tl.store(p_ht, b_h, ...)`. То есть за один проход верификации пишется `k+1` снапшот
  состояния, по одному на черновую позицию.
- `vllm/vllm/v1/attention/backends/gdn_attn.py:308-310, 329-331` —
  `spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]`:
  на запрос выдаётся ровно `num_spec + 1` слотов состояния.
- `vllm/vllm/model_executor/layers/mamba/abstract.py:74-77` —
  `num_speculative_blocks = speculative_config.num_speculative_tokens`.
- `vllm/vllm/v1/kv_cache_interface.py:695-696` — при `mamba_cache_mode` по умолчанию
  (`"none"`, задан в `vllm/vllm/config/cache.py:137`) память под состояние равна
  `page_size_bytes * (1 + num_speculative_blocks)`. Константа по S.
- `vllm/vllm/model_executor/layers/mamba/mamba_utils.py:257-268` — форма conv-состояния
  расширена до `conv_kernel_size - 1 + num_spec`: откат свёртки тоже сделан скользящим окном,
  а не пересчётом.
- `vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1443-1463` — единственный
  вызов на пути spec-декода: `fused_sigmoid_gating_delta_rule_update(..., initial_state=ssm_state,
  ssm_state_indices=spec_state_indices_tensor, num_accepted_tokens=num_accepted_tokens)`.
  Ветка `chunk_gated_delta_rule` (строка 1506, префилл) на этом пути не вызывается:
  при чистом spec-декоде `num_prefills == 0` (`gdn_attn.py:294-317`).
- Слово «recompute» на путях GDN + spec в vLLM не встречается (grep по
  `gdn_attn.py`, `qwen_gdn_linear_attn.py`, `llm_base_proposer.py`).

### Где на самом деле линейная стоимость — полное внимание в шаге верификации

- `vllm/vllm/v1/attention/ops/triton_unified_attention.py:1041-1050` — дословно:
  ```python
  use_3d = not (
      seq_threshold_3D is None
      ...
      or max_seqlen_q > 1
      or num_seqs > seq_threshold_3D
      or is_batch_invariant
  )
  ```
  `max_seqlen_q > 1` — то есть **любой** шаг спекулятивной верификации — выключает 3D-ядро.
- `.../triton_unified_attention.py:1077-1082` — сетка:
  `grid = (total_num_q_blocks, num_kv_heads)` для 2D против
  `grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)` для 3D.
- `vllm/vllm/v1/attention/backends/triton_attn.py:56` — `NUM_PAR_SOFTMAX_SEGMENTS = 16`.
  Это те самые 16 сегментов KV, которые теряются.
- `.../triton_unified_attention.py:932-935, 966` — `BLOCK_M = 16` при
  `num_queries_per_kv = 16/2 = 8`, значит `BLOCK_Q = 2`,
  `total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs`.
- `vllm/vllm/platforms/cuda.py:157-163` — приоритет бэкендов для `device_capability.major == 12`
  (sm_120): `FLASH_ATTN, FLASHINFER, TRITON_ATTN, ...`. По умолчанию на 5090 берётся FA.
- `vllm/vllm/v1/attention/backends/fa_utils.py:92-100` — на sm_120 (`major == 12`) выбирается
  `fa_version = 2`. `vllm/vllm/v1/attention/backends/flash_attn.py:176-183` — FA поддерживает
  `head_size == 256`, так что откат на Triton не происходит.
- `vllm/vllm/vllm_flash_attn/flash_attn_interface.py:311-312` —
  `if num_splits > 1: raise NotImplementedError("FA2 does not support num_splits > 1")`.
  На FA2 запросить разбиение по KV нельзя вообще.
- `vllm/vllm/v1/attention/backends/flash_attn.py:341-350` (комментарий самих разработчиков) —
  «For FA2, a graph is captured with max_query_len=1 ... This is due to special max_query_len=1
  packed-GQA handling in FA2». То есть быстрый путь FA2 привязан к `max_query_len == 1`.
- `vllm/vllm/v1/attention/backends/flash_attn.py` — **не вызывает**
  `_init_reorder_batch_threshold` вообще; то же у `triton_attn.py`. Механизм
  «spec-as-decode» (`vllm/vllm/v1/attention/backend.py:703-727`) объявляют только
  FlashInfer (`flashinfer.py:854-861`), HPC, ROCm AITER и MLA-бэкенды. Для плотного
  внимания на 5090 по умолчанию его нет.
- `vllm/vllm/v1/attention/backends/flashinfer.py:850-858` — единственный работающий на 5090
  выход: `use_dedicated_xqa = is_device_capability_family(120) and ... XQA`, и тогда
  `supports_spec_as_decode = True`. Но FLASHINFER на sm_120 не первый в списке приоритетов.

### Откат KV для 10 слоёв полного внимания

- `vllm/vllm/v1/core/sched/scheduler.py:1836-1848` — `num_accepted = max(len(generated_token_ids)
  - num_sampled, 0)`, `num_rejected = num_draft_tokens - num_accepted`, затем
  `request.num_computed_tokens -= num_rejected`. Отката как операции нет: курсор записи
  отодвигается назад, KV отвергнутых черновиков остаётся в блоках и перезаписывается
  следующим раундом. Стоимость O(1), блоки не освобождаются и не обнуляются.

### Черновая голова и логиты

- `vllm/vllm/model_executor/models/qwen3_5_mtp.py:117-123` — черновой слой MTP:
  `Qwen3_5DecoderLayer(vllm_config, layer_type="full_attention", ...)`. Черновик — это
  один слой **полного** внимания, GDN у черновика нет, откатывать там нечего.
- `vllm/vllm/model_executor/models/qwen3_5_mtp.py:240-251` — `ParallelLMHead(config.vocab_size,
  ...)` и `LogitsProcessor(config.vocab_size)`. Черновая голова — на полный словарь.
  `draft_id_to_target_id` у `Qwen3_5MTP` не определён, значит в
  `vllm/vllm/model_executor/models/interfaces.py:1487-1490` ветка `d2t` мертва.
- `vllm/vllm/v1/spec_decode/llm_base_proposer.py:436-446` — жадный черновик:
  `self.model.compute_logits(hidden_states).argmax(dim=-1)`. Полная матрица логитов
  материализуется на каждом черновом шаге; аргмакс — на устройстве.
- `vllm/vllm/v1/spec_decode/vocab_mapping.py:105-119` — `VocabMapping` строит
  `draft_to_target` / `target_to_draft` длиной в словарь, но это механизм для **разных
  токенизаторов** (гетерогенный черновик), а не для урезанной MTP-головы; для Qwen3.5/3.6
  он не включается (`speculative.py:1410-1415` требует равенства размеров словарей).
- `vllm/vllm/v1/sample/rejection_sampler.py:453-469` — жадный приём полностью на устройстве:
  `target_argmax = target_logits.argmax(dim=-1)`, затем
  `rejection_greedy_sample_kernel[(batch_size,)]`, и при `all_greedy` — ранний возврат без
  softmax. Хостовых синхронизаций в решении о приёме нет.
- `.../rejection_sampler.py:472` — недетерминированный путь: `target_probs =
  target_logits.softmax(dim=-1, dtype=torch.float32)`, то есть материализация
  `[k+1, 248320]` в fp32 (≈4 МБ при mtp3) плюс столько же на `draft_probs` за раунд.
- `.../rejection_sampler.py:271` — `output_token_ids.cpu().numpy()` — одна синхронизация
  на шаг, но только для выдачи токенов, не для приёма.

### SGLang

- `sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py:503, 515-521, 618-686` —
  явный режим `is_target_verify` с тремя протоколами: `_replayssm_fold_target_verify`
  (кольцо записей + свёртка при коммите), `_replayssm_target_verify` (циклическое кольцо) и
  запасной снапшотный `kernel_dispatcher.target_verify`.
- `sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:220-241` — ключевое
  отличие от vLLM: `disable_state_update=True`, снапшоты пишутся в **отдельный**
  `intermediate_states_buffer`, постоянное состояние во время верификации не трогается,
  и передаётся `retrieve_parent_token` — то есть цепочка состояний может идти по дереву,
  а не только по линии.
- `sglang/python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py:676-733` —
  `scatter_mamba_states_after_mtp_verify`: после решения о приёме один fused gather-scatter
  переносит выбранный промежуточный ssm-снапшот и conv-окно в постоянный кеш.
- `sglang/python/sglang/srt/mem_cache/memory_pool.py:702-716` — `intermediate_ssm_state_cache`
  формы `[num_layers, spec_state_size+1, draft_tokens, HV, K, V]`; собственный комментарий
  авторов (строки 692-701): «the dominant spec scratch, ~46x the conv state», «~9GB @ K3
  dspark γ=7». Под `--enable-linear-replayssm-spec` буфер не выделяется вовсе (`None`).
- `sglang/python/sglang/srt/layers/attention/flashattention_backend.py:1504-1510, 1710-1717` —
  `flash_attn_with_kvcache(..., max_seqlen_q=max_seqlen_q, ..., num_splits=self.num_splits)`:
  SGLang передаёт `num_splits` **на всех** путях, включая верификацию с `max_seqlen_q > 1`.
  Это и есть структурная разница с vLLM, где `num_splits` для FA2 запрещён, а на Triton
  выключается условием `max_seqlen_q > 1`.
- `sglang/python/sglang/kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py:1-47` —
  «ReplaySSM Part A», порт из vLLM: обычный декод линейного внимания без записи состояния —
  «per-step state traffic drops from read+write (~8·d·n) to read-only (~4·d·n) → roughly
  halved», флаш раз в L шагов (`linear_replayssm_cache_len` по умолчанию 16).

## Арифметика

Геометрия из `vllm/vllm/transformers_utils/configs/qwen3_5_moe.py:46-65, 95-101`:
40 слоёв, `full_attention_interval = 4` → 10 полного внимания + 30 GDN; 16 Q-голов / 2 KV-головы,
head_dim 256; GDN 16 k-голов / 32 v-головы, 128×128.

**KV на токен (bf16, 10 слоёв):** `10 × 2 × 256 × 2 × 2 B = 20 480 B = 20 KiB`.
При 200k → 4.10 ГБ; один проход на пике HBM 1792 ГБ/с = **2.29 мс**.

**Базовый декод (без MTP), из таблицы отчёта:** 1k → 3.40 мс/шаг, 200k → 5.46 мс/шаг.
Разница 2.06 мс на 199k токенов ⇒ 4.08 ГБ за 2.06 мс ⇒ **≈1980 ГБ/с**, то есть внимание
базового декода идёт **на пике HBM** (превышение над 1792 — L2 и погрешность).

**Цена раунда MTP.** При доле приёма 0.85 в раунде принимается 1.85 (mtp1), 2.57 (mtp2),
3.18 (mtp3) токена. Время раунда = принято / (ток/с):

| контекст | mtp1 | mtp2 | mtp3 |
|---|---:|---:|---:|
| 1k | 5.29 мс | 5.67 мс | — |
| 8k | 9.49 | 10.00 | 10.93 |
| 30k | 21.02 | 22.35 | — |
| 100k | — | 53.5 | — |
| 200k | — | 107.1 | — |

Наклон: mtp1 (1k→30k) 0.542 мс/1k; mtp2 (1k→200k) **0.510 мс/1k**; mtp3 (2k→16k) 0.539 мс/1k.
**Наклон не зависит от k** — значит линейная цена платится один раз за раунд, а не за
черновой шаг. Это исключает «внимание черновика по каждому шагу» и указывает ровно на
один проход верификации.

0.52 мс на 1000 токенов ⇒ 0.52 мкс на токен контекста ⇒ `20 480 B / 0.52 мкс` = **39 ГБ/с
= 2.2 % от пика HBM**. Базовый декод на тех же байтах даёт ~100 % пика. **Отношение ≈50×.**

Сетка Triton в этом режиме: `num_queries_per_kv = 8`, `BLOCK_M = 16`, `BLOCK_Q = 2`.
- обычный декод (1 строка запроса): 3D, `grid = (1, 2, 16)` = **32 CTA**, каждая читает 1/16 KV;
- mtp2 (3 строки): 2D, `total_num_q_blocks = 3//2 + 1 = 2`, `grid = (2, 2)` = **4 CTA**;
- mtp3 (4 строки): `grid = (3, 2)` = **6 CTA**, каждая стримит **весь** KV.

4–6 CTA на 170 SM — 2.4–3.5 % занятости. Измеренные 2.2 % пика HBM попадают в это ровно.
Для FA2 картина того же класса: сетка `(num_m_blocks, batch, num_heads) = (1, 1, 16)` = 16 CTA
без разбиения по KV (это следует из `num_splits > 1 → NotImplementedError` и комментария о
привязке упакованного GQA к `max_query_len == 1`; сам C++ FA2 в клоне отсутствует, так что
конкретное число CTA для FA2 — вывод, а не цитата).

**Цена снапшотов GDN (ответ на «сколько стоит один отказ»).**
Одно состояние: `32 голов × 128 × 128 = 524 288` элементов; fp32 → 2 MiB на слой;
30 слоёв → **60 MiB на снапшот на запрос**.
- За шаг верификации vLLM пишет `k+1` снапшотов и читает 1: при mtp3 это `5 × 60 MiB = 300 MiB`
  ⇒ 0.167 мс на пике HBM. Против раунда 107 мс на 200k — **0.16 %**; против 5.7 мс на 1k — 2.9 %.
- **Один отказ стоит ноль дополнительных байт и ноль дополнительного времени.** Меняется
  только индекс `i_t = num_accepted - 1` в `fused_sigmoid_gating.py:106`.
- Память: `(1 + num_spec) × 60 MiB` на конкурентный запрос; mtp3 → 240 MiB.

**Контрфакт: если бы пересчёт всё-таки был.** Работа GDN на токен на слой ≈ `4 × 32 × 128 × 128`
MAC ≈ 4.2 MFLOP; × 30 слоёв = 126 MFLOP/токен. Пересчёт от начала последовательности на 200k
= 25.2 PFLOP; даже на полном ярусе bf16 (255.3 TFLOPS) это **99 секунд**. Измеренный раунд —
0.107 с. **Гипотеза «пересчитывает от начала последовательности» опровергается на три порядка
арифметикой, независимо от кода.** Пересчёт «от последней контрольной точки» тоже исключён:
при `mamba_cache_mode="none"` (умолчание) контрольных точек по позициям не существует вовсе,
а режим `"all"` для этой модели запрещён явно —
`vllm/vllm/model_executor/models/qwen3_5_mtp.py:225-229`.

## Что у нас сегодня

**Откат состояния GDN — строго лучше vLLM.**
- `ninfer/include/ninfer/ops/gdn_replay.h:19-43` — `gdn_replay_fold` «consumes raw key/value/
  {g,beta} records in order, writes the final FP32 recurrent state»: хранятся не полные
  снапшоты состояния, а сырые записи ранга 1 на колонку, и свёртывается только принятый
  префикс `[0, commit_columns)`.
- `ninfer/src/ops/linear_attention/gated_delta_net/launch.h:41-49` —
  `launch_recurrent_record(..., const Tensor& ssm_states, ...)`: на проходе верификации
  состояние **константно**, за колонку пишутся только записи; `launch_replay_fold` пишет
  состояние ровно один раз на коммит. То есть у нас уже есть то, что SGLang включает
  флагом `--enable-linear-replayssm-spec`, а vLLM не имеет вовсе.
- Память под записи вместо снапшотов: запись на колонку — `k(16×128) + v(32×128) + {g,beta}`
  ≈ 12.3 КБ bf16 на слой против 2 MiB полного состояния, то есть **≈170× меньше**;
  ср. собственную оценку SGLang «~46x the conv state» и «~9 ГБ».
- Решение о приёме на устройстве: `ninfer/src/ops/kernel/mtp_round.cuh:10-42`
  (`mtp_prepare_next_round_kernel`) и `ninfer/src/ops/kernel/speculative_round.cuh:60-75`
  (жадный коммит «bit-identical to the original argmax accept»). Хостовых синхронизаций нет.

**Внимание в верификации — механизм болезни vLLM у нас невозможен.**
- `ninfer/src/ops/softmax_attention/dense/causal_cache/small_t.cuh:1-7` — заголовок дословно:
  «split-KV causal small-T attention». Многотокенный путь **сохраняет** разбиение по KV.
- `ninfer/src/ops/softmax_attention/dense/causal_cache/small_t.cu:99` — сетка:
  `dim3 grid(Geometry::KVHeads, splits, invocation.batch_size)`. Число `splits` берётся из
  окна KV, а **не** из числа токенов запроса.
- `.../small_t.cu:25-42` — политика: минимум `4 × SmallTSplitScale` сплитов; ярусы 64/128/256/480
  ключей на сплит по мере роста окна; потолок `SmallTMaximumSplits`.
- `ninfer/src/ops/softmax_attention/dense/causal_cache/geometry.cuh:12, 16` —
  `SmallTMaximumSplits = 85 × SmallTSplitScale`; для нашей геометрии
  `CausalD256H16Kv2 = <16, 2, 2>` это **170 сплитов**, сетка `2 × 170 = 340 CTA` на 170 SM
  (ровно две волны) при любом `T ∈ [1,6]`.
- `ninfer/src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:20-21, 327` —
  `kSmallTChunkTokens = 6`, `kMaximumVerifyTokens = 16`; при `width ≤ 6` берётся маршрут
  `SmallT`. mtp3 даёт `T = 4` ⇒ **один** проход по KV с 340 CTA. Это ровно то, что vLLM
  теряет.

**Черновая голова — у нас уже урезанная.**
- `ninfer/src/targets/qwen3_6_35b_a3b/impl/variant.h:36` — `draft_head_rows = 131072`
  (то же в `qwen3_6_27b/impl/variant.h:38`).
- `ninfer/src/targets/qwen3_6/impl/runtime/layouts_impl.h:343-347, 549-553` — при
  `ProposalHead::Optimized` под черновые логиты выделяется матрица
  `BF16 × draft_head_rows × columns`, при `Full` — `TextConfig::output_rows`.
- `ninfer/src/targets/qwen3_6/impl/runtime/dflash_impl.h:315-322` — `ops::linear` в
  `[131072, k]` BF16, `ops::argmax` на устройстве, затем `ops::proposal_remap_token_ids`
  (эквивалент `d2t`). Ни fp32-softmax, ни материализации полного словаря на черновом шаге.

**Не проверено:** поведение маршрута `ChunkedSmallT` (T = 7…16) на контекстах ≥100k
не измерялось; ниже это отдельный кандидат.

## Кандидаты для NInfer

### 1. Один проход по KV для маршрута ChunkedSmallT (T > 6)

**Механизм.** `causal_softmax_attention.cpp:288-304` (`launch_chunked_small_t`) режет окно
запроса на куски по `kSmallTChunkTokens = 6` и на каждый кусок делает **отдельный запуск
ядра**, каждый из которых сканирует весь KV: `for (begin = 0; begin < q.ne[2]; begin +=
kSmallTChunkTokens)`. При окне черновика 15 (`T = 16`) это `⌈16/6⌉ = 3` полных прохода по KV
за раунд. При окне ≤ 6 (наш канонический mtp3, T = 4) проход один — сегодня цена не видна,
потому что мы её не мерили на длинном контексте. Предлагаемая правка: перенести цикл по
кускам запроса **внутрь** CTA, чтобы одна загрузка KV-плитки обслуживала все `⌈T/6⌉` кусков
(регистровое давление растёт только по оси Q; сама split-KV сетка не меняется).

**Ожидаемый эффект — декод, длинный контекст.** KV на токен у нас (int8, брифовые 10.56 КБ):
при 200k — 2.11 ГБ, один проход на пике 1792 ГБ/с = 1.18 мс. Окно 15 сегодня платит 3 прохода
= 3.54 мс; после правки 1.18 мс, экономия **2.36 мс на раунд** = 2.11 ГБ / 1792 ГБ/с ≈ 1.3
«секунды HBM» на раунд. При 32k экономия 0.38 мс на раунд. При T ≤ 6 эффекта нет вовсе.
Это же снимает недоучёт в модели раунда `13.78 + 0.180·колонок + 0.590·шагов`: при
контекстах ≥100k член «колонки» перестаёт быть линейным и приобретает ступеньку каждые 6
колонок величиной в один проход по KV.

**Чем меряем.** Стенд: одна 5090, Qwen3.6-35B-A3B NVFP4, bs=1, greedy, каноническая точка
(чанк 8192, int8 KV). A/B: окно черновика {3, 6, 7, 15} × контекст {32k, 128k, 200k},
метрика — время раунда и время оператора `causal_softmax_attention` из профилировщика.
Предсказание до правки: между окном 6 и окном 7 при 128k+ появляется скачок ровно на
время одного прохода по KV (0.75 мс при 128k); после правки скачок исчезает.

**Риски и что ломается.** Вывод обязан остаться побитовым: изменение затрагивает только
порядок обхода, редукция сплитов не меняется — но `partial_acc/m/l` придётся расширить по оси
кусков, и объём воркспейса вырастет в `⌈T/6⌉` раз (при T=16 — втрое от текущего пика для
одного куска). Регистровое давление: q-плитка в регистрах растёт с 6 до 16 строк; на
head_dim 256 это может выбить занятость — вероятен вариант «два куска за проход» вместо всех.

**Оценка объёма.** 3–4 файла: `causal_softmax_attention.cpp` (маршрут и воркспейс),
`small_t.cu` (сетка/запуск), `small_t_i8.cuh` (+`small_t_bf16.cuh`, `small_t_fp8.cuh`) —
тело ядра.

### 2. Проверить потолок в 170 сплитов на 100–200k

**Механизм.** `small_t.cu:36-41`: выше окна 16390 целевой размер сплита — 480/SplitScale = 240
ключей, но результат зажимается в `SmallTMaximumSplits = 170` (`geometry.cuh:12`). При 200k
желаемых сплитов `⌈200000/240⌉ = 834`, фактических 170 ⇒ **1176 ключей на CTA**. Сетка
`2 × 170 = 340` CTA — две волны на 170 SM, что нормально; но ярус 480 ключей/сплит подбирался
на окнах порядка 16–32k, и на 200k хвост волны может стоить дороже, чем лишняя волна.

**Ожидаемый эффект.** Верхняя граница — доля HBM-пика на операторе внимания в декоде.
Сегодня при 200k внимание — 2.11 ГБ на шаг = 1.18 мс на пике; если фактическая доля пика
окажется, скажем, 70 %, потенциал = 0.5 мс/шаг, что при шаге ~2–3 мс есть 15–25 %. Если
доля уже >90 %, кандидат закрыт бесплатно.

**Чем меряем.** Свип операторного бенча `causal_softmax_attention` по окну
{16k, 32k, 64k, 128k, 200k} × T {1, 4} с ручным перебором `SmallTMaximumSplits`
{85·2, 128·2, 170·2 (текущее), 256·2} и ключей на сплит {240, 360, 480, 720}; метрика —
достигнутая доля от 1792 ГБ/с. **Важно:** по нашему же наблюдению об операторном стенде
(карта уходит на 180 МГц, фиксация частот запрещена) сравнивать только «та же руна против
себя» в одном прогоне, два прохода.

**Риски и что ломается.** Численность не затрагивается (split-K редукция та же), но растёт
воркспейс `partial_acc/m/l` линейно по числу сплитов: сейчас `170 × 16 голов × 256 × 4 B`
на токен = 2.8 МБ на токен; при 512 сплитах — 8.4 МБ на токен. При T=6 и batch 8 это уже
403 МБ.

**Оценка объёма.** 2 файла: `geometry.cuh` (константа), `small_t.cu` (ярусы). Плюс бенч.

### 3. (Слабый) ReplaySSM Part A для арма без спекуляции

**Механизм.** `sglang/python/sglang/kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py:1-47`
(порт из vLLM): в обычном декоде GDN состояние не записывается каждый шаг — в кольцо длиной
`L = 16` кладётся запись `(d, k, g)`, полное состояние пишется раз в L шагов, а вывод
восстанавливается из чекпойнта плюс кольцо. Трафик состояния падает с чтение+запись до
только-чтение.

**Ожидаемый эффект.** У нас `ninfer/src/ops/linear_attention/gated_delta_net/recurrent.cuh:592-609`
(`load_state_tile` / `store_state_tile`) — на пути `launch_recurrent_batch_update`
(`launch.h:34-38`) состояние читается и пишется каждый шаг. Состояние по геометрии
30 × 32 × 128 × 128 × 4 B = **62.9 MiB** (бриф считает 120 МБ). Экономия — одна запись
состояния на шаг: 62.9 MiB / 1792 ГБ/с = **35 мкс**, то есть 3.5 % от пика HBM на шаг;
при декоде 579 ток/с (1.73 мс/шаг) это **2.0 %** (по брифовым 120 МБ — 3.9 %).
**Но:** на нашей канонической точке (mtp3) декод идёт по пути `launch_recurrent_record`,
где состояние и так не пишется — выигрыш достаётся только арму mtp0, который каноническим
не является. Ставить в очередь низко.

**Чем меряем.** A/B mtp0 на контекстах {4k, 32k}: ток/с и время оператора GDN.

**Риски.** Восстановление вывода из чекпойнта + кольца — другая последовательность
операций fp32 ⇒ **вывод перестаёт быть побитовым**. По нашим правилам это переводит
кандидата в разряд «мерится только скоростью, точность отдельной батареей».

**Оценка объёма.** 2–3 файла в `gated_delta_net/`.

## Опровергнуто / не переносится

- **Гипотеза отчёта «путь верификации пересчитывает состояние, а не откатывает его» — неверна.**
  Три независимых опровержения: (а) ядро явно читает снапшот по индексу принятых токенов
  (`fused_sigmoid_gating.py:105-116`) и пишет снапшот на каждую колонку (строки 157-166);
  (б) кеш состояний выделяется с `1 + num_spec` слотами (`gdn_attn.py:308-310`,
  `abstract.py:74-77`, `kv_cache_interface.py:695-696`); (в) пересчёт от начала на 200k стоил
  бы ≈99 с против измеренных 0.107 с. Отчёт здесь противоречит коду; линейная цена лежит в
  полном внимании.
- **Снапшоты состояния (vLLM `spec_state_indices_tensor`, SGLang `intermediate_ssm`) нам не
  нужны** — у нас те же гарантии дают записи ранга 1 (`gdn_replay.h:19-43`), которые в ~170
  раз компактнее полного состояния. Заимствовать нечего; наоборот, SGLang сам вводит
  `--enable-linear-replayssm-spec`, чтобы убрать снапшоты и прийти к нашей схеме
  (`memory_pool.py:702-703`).
- **Дерево в GDN-верификации (SGLang `retrieve_parent_token`, `gdn_triton.py:240`)** — механизм
  корректный и у нас его нет, но по нашей же модели раунда `13.78 + 0.180·колонок +
  0.590·шагов` дерево на MoE не окупается. Не переносится по причине, не связанной с GDN.
- **`num_splits` как параметр внимания (SGLang `flashattention_backend.py:1504-1510`)** — это
  их обходной путь вокруг чужого ядра FA. У нас разбиение по KV не параметр вызова, а свойство
  маршрута (`small_t.cu:99`), и оно уже включено при T ≤ 6. Заимствовать нечего.
- **`mamba_cache_mode="align"/"all"` и `mamba_get_block_table_tensor`
  (`vllm/vllm/v1/attention/backends/utils.py:1004-1042`)** — это префиксное кеширование
  состояний GDN по блокам, а не спекуляция. У нас эквивалент уже есть
  (`ninfer/src/targets/qwen3_6/impl/runtime/state_image_store.h`,
  `prefix_identity.h`). Не находка.
- **`VocabMapping` (`vllm/vllm/v1/spec_decode/vocab_mapping.py`)** — механизм для черновика с
  **другим токенизатором**; для одинакового словаря vLLM его запрещает
  (`config/speculative.py:1410-1415`). К нашему случаю (одна и та же токенизация, урезанная
  голова) неприменим; наш `proposal_remap_token_ids` + `draft_head_rows = 131072`
  (`dflash_impl.h:315-322`, `variant.h:36`) решает ту же задачу и дешевле, чем полная
  голова vLLM.
- **`LocalArgmaxMixin` / `LogitsProcessor.get_top_tokens`
  (`vllm/vllm/model_executor/models/interfaces.py:1458-1490`,
  `vllm/vllm/model_executor/layers/logits_processor.py:189-234`)** — экономия только на
  коммуникации между TP-рангами (`O(batch·2·tp_size)` вместо `O(batch·vocab)`). При TP=1
  вырождается в обычный argmax (`logits_processor.py:225-226`). Вне области действия брифа.
- **Откат KV полного внимания у vLLM (`scheduler.py:1846`)** — тот же приём, что у нас:
  курсор назад, блоки не трогаются. Платы там нет ни у них, ни у нас. Проверено, ничего не
  найдено.
- **fp32-softmax по полному словарю в отбраковщике (`rejection_sampler.py:472`)** — 4 МБ за
  раунд при mtp3, ≤0.005 мс. На фоне 107 мс раунда шум; не объясняет ничего и заимствовать
  нечего (наш `speculative_round.cuh` и так на устройстве).
- **ReplaySSM Part A** — см. кандидат 3: механизм реален, но бьёт по арму, который у нас не
  канонический, и ломает побитовость. Оставлен в списке только ради полноты.

## Открытые вопросы

1. Каким бэкендом внимания снят замер из отчёта — FLASH_ATTN (умолчание на sm_120) или
   TRITON_ATTN? Для Triton вывод доказан построчно (`use_3d` при `max_seqlen_q > 1`);
   для FA2 доказано только, что `num_splits > 1` запрещён и что упакованный GQA привязан к
   `max_query_len == 1`, а само C++-ядро FA2 в клоне отсутствует. Проверка на их стороне
   тривиальна: `VLLM_ATTENTION_BACKEND=FLASHINFER` с XQA-ядром декода включает
   `supports_spec_as_decode` (`flashinfer.py:850-858`) — если наклон 0.51 мс/1k при этом
   исчезает, диагноз подтверждён полностью.
2. Наша модель раунда `13.78 + 0.180·колонок + 0.590·шагов` снималась на коротком контексте.
   Ступенька каждые 6 колонок (маршрут `ChunkedSmallT`) в ней не учтена. Нужно ли
   перефитить модель с членом `⌈колонок/6⌉ × время_прохода_KV(S)` — вопрос к тому, будем ли
   мы вообще пользоваться окнами >6 на длинном контексте.
3. Достигаемая доля HBM-пика оператором `causal_softmax_attention` при T=4 и окне 100–200k
   у нас не измерена. Без неё кандидат 2 остаётся гипотезой.
4. Наклон 0.52 мс/1k у vLLM объяснён с точностью ~10 % (2.2 % пика против расчётных 2.4–3.5 %
   занятости). Остаточные ~10 % могут быть вторым эффектом (например, тем, что 2D-ядро
   читает тайл TILE_SIZE_PREFILL вместо TILE_SIZE_DECODE — `triton_unified_attention.py:1079,
   1082`); не разбиралось.
