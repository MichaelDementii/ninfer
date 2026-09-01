# V3 — путь FP8 KV-кеша в vLLM

Клоны: `vllm` (v0.28.0, HEAD `2cf0a69`), `flashinfer`, `ninfer` (master `da49c0d`).
Все ссылки — на локальные клоны. Сборка/запуск не производились.

---

## Что это за механизм

`--kv-cache-dtype fp8` в vLLM — это **только формат хранения**, самый примитивный из
семейства: один статический скаляр на весь тензор K и один на V, взятый из чекпоинта,
а если его там нет — **1.0** с предупреждением в лог. Квантование делает отдельное ядро
`reshape_and_cache_flash`, которое и так вызывается каждый шаг для записи K/V в страницы;
fp8 меняет в нём только тип назначения и добавляет умножение на обратный скейл.
Раскладка страницы не меняется вообще — та же `(num_blocks, block_size, num_kv_heads,
2*head_size)`, только `itemsize` 1 вместо 2.

Интереснее то, что рядом: в этом же дереве живут ещё пять режимов KV-квантования
(`int8/fp8/int4_per_token_head`, `nvfp4`, `nvfp4_4over6`, четыре `turboquant_*`), и вот
у них уже есть механика — динамические скейлы на (токен, голову), скейлы **внутри той же
страницы**, поиск скейла по MSE, вращение Адамара, кодовая книга Ллойда—Макса.

**Главный вывод по загадке префилла: +43% на 10k — это не байты KV и не планировщик.
Флаг `--kv-cache-dtype fp8` на sm_120 молча меняет бэкенд внимания
`FLASH_ATTN` → `FLASHINFER`, потому что FlashAttention отказывается работать с fp8-кешем
где угодно кроме FA3/sm90 и FA4/sm100.** Разбор всех четырёх гипотез — ниже.

---

## Доказательства из кода

### Путь байта K и V (вопрос 1)

- `vllm/csrc/libtorch_stable/cache_kernels.cu:264` — `reshape_and_cache_flash_kernel`,
  единственное место, где K/V попадают в кеш. Отдельное ядро, не эпилог проекции,
  не часть ядра внимания. Один и тот же вызов на префилле и на декоде.
- `vllm/csrc/libtorch_stable/cache_kernels.cu:190-201` — `CopyWithScaleOp`: при
  `kv_dt == kAuto` просто `static_cast`, иначе `fp8::scaled_convert<OutT, InT, kv_dt>(src, scale)`.
- `vllm/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:559` — `scaled_convert`,
  через `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)` (`:214-215`). Насыщение,
  не NaN.
- `vllm/csrc/libtorch_stable/cache_kernels.cu:303-312` — быстрый путь NHD + скейл `[1]`:
  скейл читается один раз на CTA, дальше векторизованная копия `vectorize_with_alignment<8>`.
- `vllm/csrc/libtorch_stable/cache_kernels.cu:805-807` — единственные допустимые формы
  скейла: `numel() == 1` **или** `numel() == num_heads`. То есть максимум — **на голову**.
  Ни на токен, ни на блок, ни на группу каналов.
- `vllm/model_executor/layers/quantization/kv_cache.py:126-129` — и даже форма
  `[num_heads]` для обычного fp8 отсекается на уровне Python:
  `raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")`.
- `vllm/model_executor/layers/quantization/kv_cache.py:105-108` — если скейлов в чекпоинте
  нет, `k_scale = v_scale = 1.0`.
- `vllm/model_executor/layers/quantization/kv_cache.py:145-151` — предупреждение
  «Using KV cache scaling factor 1.0 for fp8_e4m3… verify that k/v_scale scaling factors
  are properly set in the checkpoint».
- Динамической калибровки в рантайме **нет**: `--calculate-kv-scales` вырезан, во всём
  дереве остался один комментарий (`vllm/models/kimi_k3/nvidia/mla.py:445`).

### Что ядро внимания делает с fp8-кешем (вопрос 3), на sm_120

- `vllm/v1/attention/backends/fa_utils.py:223-243` — `flash_attn_supports_kv_cache_dtype`
  возвращает `True` **только** для `(fa_version == 3 and is_device_capability_family(90))`
  или `(fa_version == 4 and is_device_capability_family(100))`.
- `vllm/v1/attention/backends/fa_utils.py:92-100` — на всём, что не sm90/sm100,
  `fa_version = 2`. Значит на sm_120 условие ложно всегда.
- `vllm/v1/attention/backends/flash_attn.py:224-234` — `supports_combination` возвращает
  дословно `"FP8 KV cache requires FA3 on SM90 or FA4 on SM100"`.
- `vllm/platforms/cuda.py:156-163` — приоритет бэкендов на `device_capability.major == 12`,
  не-MLA: `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]`.
- `vllm/v1/attention/backends/flashinfer.py:408-418` — FlashInfer объявляет `fp8`,
  `fp8_e4m3`, `fp8_e5m2`, `nvfp4`; `:527-529` — головы `[64, 128, 256, 512]`;
  `:532-540` — capability `[8.0 … 12.1]`. Все три условия для нашей конфигурации проходят.
- Итог: **bf16 → `FLASH_ATTN` (FA2), fp8 → `FLASHINFER`.** Это не вариант ядра, это другой
  бэкенд с другим ядром префилла и другим ядром декода.
- `vllm/utils/flashinfer.py:410-417` — на sm12x у FlashInfer есть **только XQA-декод**,
  trtllm-префилла нет (`return not is_prefill`).
- `vllm/utils/flashinfer.py:516-521` — при sm12x + fp8 декод принудительно уходит на XQA.
- `vllm/v1/attention/backends/flashinfer.py:950` и `:961` — Q остаётся в bf16: комментарий
  прямо говорит «Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries».

Что делает само ядро FlashInfer с fp8 на sm_120 (FA2, `prefill.cuh`):

- `flashinfer/include/flashinfer/attention/prefill.cuh:330-332` — `USE_KV_REPACK` включается
  при `sizeof(DTypeKV)==1 && HEAD_DIM_VO != 64 && HEAD_DIM_VO <= 256 && CTA_TILE_Q > 16`.
- `flashinfer/include/flashinfer/attention/prefill.cuh:1186-1211` — `repack_fp8_tile_to_bf16`:
  **весь K-тайл и весь V-тайл распаковываются в bf16 в отдельный буфер shared memory**
  (`KVRepackSmem`, `:156-160`), с `block.sync()` до MMA (`:3099-3103`, `:3162-3166`).
- `flashinfer/include/flashinfer/attention/prefill.cuh:1289-1293`, `:1823-1827` — MMA всегда
  `mma_sync_m16n16k16_row_col_f16f16f32`. **FP8-MMA в `prefill.cuh` нет вообще.**
- `flashinfer/include/flashinfer/vec_dtypes.cuh:218-259` — распаковка fp8→bf16 идёт
  программной последовательностью `__byte_perm` + `lop3` (аппаратный `cvt` есть только
  для `half`, `flashinfer/csrc/xqa/utils.cuh:727-753`).
- `flashinfer/include/flashinfer/mma.cuh:217` — `mma_sync_m16n16k32_row_col_f8f8f32`
  (настоящий `m16n8k32.e4m3.e4m3`) **определён и нигде не вызывается**; единственные живые
  потребители fp8-MMA на sm120 — sparse-MLA DSA-ядра
  (`flashinfer/include/flashinfer/attention/sparse_mla_sm120/arch/mma_sm120.cuh:50`).
- Настоящий fp8-путь на тензорных ядрах есть только у FA3/sm90:
  `flashinfer/include/flashinfer/attention/hopper/quantization/kernel_traits.cuh:199,203-205`
  (`wgmma` e4m3 для QK **и** для PV), при этом `flashinfer/flashinfer/utils.py:467-472`
  требует, чтобы **Q тоже был fp8**.
- `flashinfer/flashinfer/utils.py:527-546` — специализированные sm120-ядра `fmha_v2_sm120`
  **исключают fp8** (`dtype_kv in {float16, bfloat16}`). То есть fp8 на 5090 отбрасывает
  и этот быстрый путь тоже.

### Раскладка кеша (вопрос 2)

- `vllm/v1/kv_cache_interface.py:243-259` — вся арифметика страницы:
  `state_content_size_bytes = (head_size + head_size_v) * get_dtype_size(dtype)`,
  `unpadded_page_size_bytes = num_heads * block_size * state_content_size_bytes`.
- `vllm/utils/torch_utils.py:39-41` — `fp8`/`fp8_e4m3`/`fp8_e5m2` → `torch.uint8`, itemsize 1.
  Отсюда ровно 2× по ёмкости и ноль изменений в форме.
- `vllm/v1/attention/backends/flash_attn.py:159-172` — форма
  `(num_blocks, block_size, num_kv_heads, 2*head_size)` (NHD) или
  `(num_blocks, num_kv_heads, block_size, 2*head_size)` (HND). **K и V упакованы в одну
  ось контента**, не два тензора.
- Отдельно, режим per-token-head — вот здесь механика есть:
  `vllm/v1/attention/backends/triton_attn.py:351-369` — головной размер добивается на
  `sizeof(float32)/sizeof(cache_dtype)` элементов, и **скейл fp32 лежит внутри той же
  страницы, сразу за данными этой (головы, токена)**:
  `[K(hs) | K_scale(pad) | V(hs) | V_scale(pad)]`
  (`vllm/v1/attention/backends/triton_attn.py:440-501`, `_ensure_scale_caches` строит
  strided-вью fp32 поверх этих байт).
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:199-247` — само динамическое
  квантование: `k_scale = max(max(abs(k_h)) / QUANT_MAX, 1e-6)` по head_size, одна голова
  на программу, round-half-away-from-zero перед усечением. `QUANT_MAX` параметризован:
  127 для int8, 448 для fp8 (`:262-266`).
- NVFP4-KV: `vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:7-9` — раскладка
  `[K_data | K_scale | V_data | V_scale]`, данные и скейлы — **раздельные непрерывные
  регионы на голову**, чтобы дескриптор TMA ложился напрямую; `:33-47` — swizzle скейлов
  под trtllm-gen на SM100.

### Взаимодействие с планировщиком (вопрос 4) — гипотеза (а)

- `vllm/config/scheduler.py:42` — `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`; `:74` —
  `enable_chunked_prefill = True`; `:70` — `long_prefill_token_threshold = 0`.
- `vllm/engine/arg_utils.py:2602-2611` — реальный дефолт по объёму VRAM: для карты
  «всё остальное» (32 ГБ, RTX 5090) это **8192 для LLM-класса и 2048 для OpenAI-API-сервера**.
  Отсюда и берётся отдельный эффект `--max-num-batched-tokens 8192`.
- `vllm/engine/arg_utils.py:2572` — вход в эту эвристику `get_device_total_memory()`,
  **полный объём платы**, а не свободная память под кеш. Размера кеша на этом этапе ещё
  не существует: `cache_config.num_gpu_blocks` заполняется позже
  (`vllm/v1/engine/core.py:319`, `vllm/v1/worker/gpu_worker.py:670`).
- Ширина чанка: `vllm/v1/core/sched/scheduler.py:966` (waiting) и `:565-567` (running) —
  `min(max_num_batched_tokens, long_prefill_token_threshold or ∞, остаток)`. Числа блоков
  в этих выражениях нет.
- `vllm/v1/core/sched/scheduler.py:171-172` — `num_gpu_blocks` читается, проверяется
  `> 0` и **больше нигде не используется**.
- `vllm/v1/core/kv_cache_manager.py:475-491`, `:526-530` — ёмкость влияет только бинарно:
  `allocate_slots` возвращает `None`. Планировщик на это либо вытесняет чужой запрос и
  повторяет с **той же** шириной (`scheduler.py:643-685`), либо ждёт (`:1033-1054`).
  Ширину он не уменьшает никогда.
- `vllm/config/cache.py` — `cache_dtype` не участвует ни в одном выражении
  `SchedulerConfig`. Грепом по `vllm/config/*.py` связок нет.
- CUDA-графы: `vllm/config/vllm.py:1940-1945` — потолок захвата
  `min(max_num_batched_tokens, min(max_num_seqs*(1+spec)*2, 512))`; и 2048, и 8192
  дают одинаковые 512, префилл-батчи в граф не попадают
  (`vllm/config/compilation.py:615,630-632` — `FULL_AND_PIECEWISE`, полные графы только
  для декод-батчей).

**Вывод по (а): опровергнуто. Запрос 10k при `max_num_batched_tokens=2048` идёт
`ceil(10000/2048) = 5` шагов при любой ёмкости кеша.**

### Точность (вопрос 5)

- `vllm/config/cache.py:19-37` — весь список `CacheDType`; `fp8` = `fp8_e4m3`, `fp8_e5m2`
  есть отдельно.
- `vllm/model_executor/layers/attention/attention.py:208` —
  `"fp8_e5m2 kv-cache is not supported with fp8 checkpoints."`
- `vllm/v1/attention/backends/fa_utils.py:231-232` — FlashAttention отказывает `fp8_e5m2`
  безусловно, ещё до арх-гейта. То есть e5m2 существует, но почти нигде не поддержан.
- Выбросы: никакой обработки. Только `__NV_SATFINITE` в конверсии
  (`vllm/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:214-215`) и статический
  скейл на тензор.
- `vllm/config/cache.py:114` + `vllm/model_executor/layers/attention/attention.py:290-306` —
  `--kv-cache-dtype-skip-layers`: **послойное исключение из квантования** по индексу слоя
  или по типу внимания (`sliding_window`); исключённый слой получает `kv_cache_dtype = "auto"`.
- `vllm/engine/arg_utils.py:2018-2026` — для `turboquant_*` граничные слои добавляются
  в этот список автоматически.
- `vllm/model_executor/layers/quantization/turboquant/config.py:190-192` — почему это важно:
  «Empirically required for aggressive presets — without it GSM8K drops ~30 points
  on Qwen3-4B».
- `vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:118-160` — `nvfp4_4over6`:
  считаются **два кандидата скейла** (`max/6` и `max/4`), для каждого — L2-ошибка
  реконструкции (`nvfp4_reconstruction_error`, `:89-115`, с точной эмуляцией
  `cvt.rn.satfinite.e2m1x2` в `round_to_nearest_e2m1`, `:50-71`), берётся меньшая.
  Всё в регистрах, данные уже загружены.
- `vllm/config/cache.py:84-85` — документировано как «selects between max/6 and max/4
  scales per 16 values by minimizing squared reconstruction error».

### TurboQuant (вопрос 6)

- `vllm/v1/attention/backends/turboquant_attn.py:179-210` — `get_kv_cache_shape` =
  `(num_blocks, num_kv_heads, block_size, slot_size_aligned)`, **без ведущей двойки**:
  K и V лежат в одном слоте, вперемежку со своими скейлами.
- `vllm/model_executor/layers/quantization/turboquant/config.py:20-41,71-75` — четыре пресета:
  `k8v4` (fp8-ключи + 4-битные значения, 2.6×, +1.17% PPL), `4bit_nc` (3.8×, +2.71%),
  `k3v4_nc` (+10.63%), `3bit_nc` (4.9×, +20.59%).
- Ключи (не-fp8 пресеты): нормировка на **L2-норму** (не absmax) →
  **Адамар Сильвестра** → квантование по кодовой книге **Ллойда—Макса** для `N(0, 1/d)`:
  `vllm/v1/attention/ops/triton_turboquant_store.py:414-417`,
  `vllm/v1/attention/backends/turboquant_attn.py:105-120` (H симметричен, `Pi == PiT`),
  `vllm/model_executor/layers/quantization/turboquant/centroids.py:31-79`.
- Q вращается тем же H на декоде: `vllm/v1/attention/ops/triton_turboquant_decode.py:522-528`.
- Значения — всегда **асимметричное аффинное** целочисленное квантование с zero-point
  (`triton_turboquant_store.py:101-113`), V не вращается никогда.
- Скейлы: по (токен, kv-голова), fp16, **дописаны внутрь того же слота**
  (`triton_turboquant_store.py:318-325`, `:122-135`).
- Цена: на основном CUDA-пути **тензорные ядра не используются вообще** — стадия 1
  декода деквантует в fp32-регистры и делает поэлементное умножение + `tl.sum`
  (`vllm/v1/attention/ops/triton_turboquant_decode.py:208-211`, `:304`, `num_warps=1`
  на `:586`). MMA появляется только на AMD gfx950 (FlyDSL) и в SoA-Triton-варианте.
- Префилл первого чанка идёт **по несжатым bf16 K/V через FlashAttention**
  (`turboquant_attn.py:769-778`); сжатие только на стороне записи. Продолжение —
  либо переиспользование декод-ядра при `q_len ≤ 128` (`:862-916`), либо массовый
  деквант всего контекста в fp16 + обратное вращение + FlashAttention (`:1029,1065,1098`).
- `vllm/engine/arg_utils.py:2385-2396` — включение любого `turboquant_*`
  **принудительно откатывает FlashAttention на версию 2**.
- Ссылки: DRIVE/EDEN, HIGGS (arXiv:2411.17525), arXiv:2501.19392, TurboQuant (Zandieh et al.,
  ICLR 2026) — `turboquant/config.py:51-69`. Там же прямо сказано, что QJL намеренно
  не реализован.

### Побочная находка: fp8 меняет `block_size` на гибридных моделях

Qwen3.6-35B-A3B в терминах vLLM — гибрид GDN + full attention (`Qwen3NextForCausalLM`,
`vllm/model_executor/models/qwen3_next.py:814-833`, состояние через
`MambaStateShapeCalculator.gated_delta_net_state_shape`). Для таких моделей страница
внимания принудительно подгоняется под страницу состояния GDN:

- `vllm/platforms/interface.py:840-850` — `attn_page_size_1_token =
  FullAttentionSpec(block_size=1, ..., dtype=kv_cache_dtype).page_size_bytes`,
  то есть **линейно зависит от itemsize кеша**.
- `vllm/platforms/interface.py:903-914` —
  `attn_block_size = kernel_align * cdiv(mamba_page_size, kernel_align * attn_page_size_1_token)`,
  затем `if cache_config.block_size < attn_block_size: cache_config.block_size = attn_block_size`.
- `vllm/platforms/interface.py:920-940` — страница mamba добивается до страницы внимания
  ровно (`mamba_page_size_padded = attn_page_size`).

Арифметика для 2 KV-голов × head_dim 256: `attn_page_size_1_token` = 2048 B (bf16) →
1024 B (fp8), значит **`cache_config.block_size` в токенах удваивается**. Это не влияет
на ширину чанка префилла (`mamba_cache_mode` по умолчанию `"none"`,
`vllm/config/cache.py:137`, а `need_mamba_block_aligned_split` требует `"align"`,
`vllm/v1/core/sched/scheduler.py:316-318`), но меняет геометрию страниц, которую видит
ядро внимания, и объём хвостовых потерь на неполном блоке.

---

## Что у нас сегодня

- **Место квантования.** Префилл — отдельное ядро `kv_cache_append_full_i8_kernel`
  (`ninfer/src/ops/kv_cache/append/kernel.cuh:203`), запускается прямо перед ядром внимания
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/prompt.cu:91`). Декод (T ≤ 6) —
  **сплавлено внутрь ядра внимания**
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh:209-286`).
  Развилка — `causal_softmax_attention.cpp:324-336`, `kSmallTChunkTokens = 6` (`:20`).
- **Гранулярность скейла.** По (токен, KV-голова, **группа 64 каналов**) — 4 скейла на
  токен на голову для K и 4 для V: `ninfer/src/ops/kv_cache/int8_g64_codec.cuh:19-21`,
  `ninfer/src/ops/kv_cache/d256_profile.h:22-23`. Absmax по 64 каналам, варп-редукция
  (`ninfer/src/ops/kv_cache/append/kernel.cuh:246-247`). Симметрично, без zero-point,
  коды в `[-127, 127]` (`int8_g64_codec.cuh:54,65`). Скейл хранится в FP16, а коды строятся
  по обратной величине **представленного** FP16 — кодек точно обратим (`:52-59`).
- **Где живут скейлы.** Отдельные плоскости, `FP16 [4, 64, kv_heads, pages]`
  (`ninfer/src/targets/qwen3_6/impl/state/decoder_state.cpp:41-47,104-119`),
  всё в одной арене, выравнивание 256 B. 264 B на токен на голову для K, столько же для V.
- **Раскладка.** Страница 64 токена, жёстко (`ninfer/src/core/paged_kv_cache.h:17`,
  `paged_kv_cache.cpp:39-41`). K и V — раздельные плоскости. Раскладка **одинакова**
  на префилле и на декоде (`int8_g64_codec.cuh:23-35` используется всеми путями).
- **Что делает ядро.** QK — настоящий int8-MMA
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`ninfer/src/ops/common/mma.cuh:51-57`,
  вызовы `prompt_i8.cuh:299-300`, `small_t_i8.cuh:441-442`), K **не** деквантуется.
  Q квантуется тем же G64-рецептом внутри ядра (`prompt_i8.cuh:145-172`).
  PV — V деквантуется в тайл f16/bf16 и идёт обычный `mma_f16`/`mma_bf16`
  (`prompt_i8.cuh:440-441`, `small_t_i8.cuh:600-601`). Причина сформулирована в исходнике:
  `small_t_i8.cuh:11-13` — «V is quantized per key, so its scale cannot be factored out
  of a key-contracted int8 accumulation».
- **Аккумуляторы.** Все f32 (`ninfer/src/ops/common/mma.cuh:33-103`, `prompt_i8.cuh:241`,
  `small_t_i8.cuh:349`). Свёртка в f32 — **на группу 64 каналов** в QK
  (`prompt_i8.cuh:312-315`), а не на 64 ключа.
  *Расхождение с брифом:* бриф говорит «f16 accumulation inside the int8 attention kernel
  with per-64-key folding to f32» — в клоне на `da49c0d` f16-аккумуляции нет нигде
  (грепы `.f16.f16.f16` / `.bf16.bf16.bf16` пусты); свёртка на 64 канала. Патч
  «PV fp16-acc», о котором говорит журнал экспериментов, в клон не влит.
- **Кольца свежих токенов нет** — и это записано в исходнике явно:
  `small_t_i8.cuh:14-16` «All keys (history AND the current/diagonal tokens) are read from
  the quantized cache… No from_new special-casing». Быстрый путь `from_new` есть только
  в bf16-ядре (`small_t_bf16.cuh:230-236`).
- **Адамар у нас есть.** *Второе расхождение с брифом:* ортонормальный H256 применяется
  к K перед квантованием и к Q внутри ядра внимания
  (`ninfer/src/ops/kv_cache/hadamard_d256.cuh:46-65`, `append/kernel.cuh:234`,
  `prompt_i8.cuh:156`, `small_t_i8.cuh:302`). V не вращается. Калибровки нет, клиппинга
  выбросов нет — только насыщение кода.
- **FP8-KV у нас тоже есть**, продакшн: `ninfer/src/ops/kv_cache/fp8_e4m3_row_codec.cuh`,
  группа 256 (один FP16-скейл на всю строку D256), `absmax/448`, `--kv-dtype fp8`
  (`ninfer/src/serve/serve_options.cpp:48-53`). QK там идёт на
  `mma.kind::f8f6f4.m16n8k32.e4m3.e4m3.f32` (`ninfer/src/ops/common/mma.cuh:59-66`),
  **PV — по-прежнему `mma_f16`**.
- **NVFP4-KV у нас нет** (`ninfer/src/ops/kv_cache/d256_profile.h:18-28` допускает только
  BF16/I8/FP8_E4M3FN); NVFP4 — только веса/активации.
- **Ширина чанка префилла в коде — 1024** (`ninfer/include/ninfer/types.h:116`,
  `ninfer/src/runtime/engine/engine.cpp:28`, флаг `--prefill-chunk`,
  `ninfer/apps/cli/options.cpp:132`, кратность 128 — `:206`). 8192 в исходниках нет нигде;
  рабочие 8192 задаются флагом.
- Послойного выбора формата KV **нет**: `--kv-dtype` глобальный
  (`ninfer/include/ninfer/types.h:117`, `ninfer/src/serve/serve_options.h:44`).

---

## Разбор загадки префилла

Замер: 10k префилл 22 015 → 31 415 ток/с (+43%), 30k 21 496 → 24 280 (+13%),
1k и 100k/200k без изменений (на 100k/200k fp8 даже чуть медленнее).

### (г) Артефакт замера — частично да, но не в fp8

Дефолт `max_num_batched_tokens` на 32-гигабайтной карте — **2048 для API-сервера**
и 8192 для LLM-класса (`vllm/engine/arg_utils.py:2602-2611`). Отдельная строка таблицы
«`--max-num-batched-tokens 8192` даёт +52% на 10k» объясняется полностью и тривиально:
это ровно наш собственный результат про ширину чанка и недокормленный grouped-MoE.
Совпадение величин (+43% и +52%) — совпадение: механизмы разные и **независимые**.
Если оба флага дать вместе, эффекты должны сложиться; если не складываются — значит
что-то в этом разборе неверно, и это готовый тест.

### (а) Планировщик перестаёт дробить префилл — опровергнуто

Ссылки выше. `cache_dtype` не входит ни в одно выражение `SchedulerConfig`; число блоков
входит в планировщик только как булев сигнал «влезло / не влезло». 10k при чанке 2048 —
5 шагов при любой ёмкости.

### (в) Квантование вынесено в другое место конвейера — опровергнуто

Оно и при `auto`, и при `fp8` делается одним и тем же ядром `reshape_and_cache_flash`
(`cache_kernels.cu:264`), которое при `auto` просто копирует (`CopyWithScaleOp`, `:195-196`).
fp8 **добавляет** работу (умножение + конверсия), а не убирает.

### Роуфлайн: байты KV не могут дать +43%

Модель: 10 GQA-слоёв, 2 KV-головы, head_dim 256.
KV на токен: bf16 = 10·2·256·2·2 = 20 480 B (20.0 KiB); fp8 = 10 240 B.
(Наш int8 с учётом скейлов = 10 560 B — ровно «S × 10.56 KB» из журнала. Сходится.)

- **Запись кеша** за префилл 10k: 204.8 MB → 102.4 MB. Экономия 102.4 MB при пике
  1792 GB/s — **57 мкс**. Префилл 10k при 22 015 ток/с длится 454 мс. Это **0.013%**.
- **Перечитывание кеша** ядром внимания, худший случай без L2: чанк 2048, Q-тайл 64 →
  32 тайла на чанк; Σ по 5 чанкам = 32·2048·(1+2+3+4+5) = 983 040 чтений токена
  × 20 480 B = 20.1 GB → 11.2 мс = **2.5%** префилла. Половина от этого — **≤1.2%**.
  Реально сильно меньше: KV одного слоя на 10k токенов — 20.5 MB, а L2 = 96 MB, то есть
  повторные чтения ловятся кешем.

**Потолок эффекта от байтов KV на префилле — 1.2%. Измерено +43%. Разрыв в 36 раз.**
Гипотеза «меньше байт» на префилле закрыта арифметикой.

Для контраста, **декод объясняется байтами полностью и количественно**. Пусть W — байты
не-KV трафика на шаг декода. Из точки 100k: (W+2.05)/(W+1.02) = 1.19 → W ≈ 4.4 GB.
Из точки 200k: (W+4.10)/(W+2.05) = 1.36 → W ≈ 3.6 GB. Один параметр, две независимые
точки, согласие в пределах 20%. Тот же W ≈ 4 GB предсказывает +2.5% на 10k (измерено 0%)
и +7% на 30k (измерено +3.9%). Декод — это чистая экономия байт, ничего больше.

### (б) Ядро идёт другим маршрутом — подтверждено, и это ответ

`--kv-cache-dtype fp8` на sm_120 **меняет бэкенд**:

| | bf16 (`auto`) | `fp8` |
|---|---|---|
| бэкенд | `FLASH_ATTN` | `FLASHINFER` |
| префилл | vllm-flash-attn FA2, varlen paged | FlashInfer FA2 `prefill.cuh` + распаковка fp8→bf16 в smem |
| декод | vllm-flash-attn FA2 paged | XQA (`csrc/xqa/mha.cu`), bf16 Q, `convertKCacheWordToF16` |
| MMA | `m16n16k16 f16f16f32` | **тот же** `m16n16k16 f16f16f32` |

Число MMA-инструкций идентично (`prefill.cuh:1229-1305`, `:1770-1842` — циклы не зависят
от `DTypeKV`). Более того, fp8 на этом пути **добавляет** работу: лишний проход
smem→конверсия→smem на каждый K- и V-тайл плюс два `block.sync()`
(`prefill.cuh:1199-1211`, `:3099-3103`, `:3162-3166`), причём конверсия в bf16 — программная
(`vec_dtypes.cuh:218-259`).

Отсюда единственное непротиворечивое чтение таблицы: **ядро префилла FlashInfer при
head_dim 256 быстрее ядра vllm-flash-attn FA2 примерно на 43%, и этот выигрыш съедается
налогом на распаковку по мере роста контекста** — налог растёт как объём перечитываемого
KV, то есть быстрее, чем выигрыш. На 1k префилл слишком короткий, чтобы что-то показать;
на 10k выигрыш максимален; на 30k уже наполовину съеден; на 100k/200k налог перевешивает
и знак меняется (12 865 → 12 580, 7 862 → 7 447). Форма кривой воспроизводится.

**Решающий A/B для оператора (один прогон, ничего не меняя в модели):**
запустить конфигурацию по умолчанию (bf16-KV) с `VLLM_ATTENTION_BACKEND=FLASHINFER`.
Если +43% на 10k появляется без всякого fp8 — вопрос закрыт, и в таблице надо править
подпись: это строка про бэкенд, а не про формат кеша. Заодно стоит просто прочитать в
логе строку «Using … backend» в обоих прогонах — vLLM её печатает, и она сразу покажет
подмену.

---

## Кандидаты для NInfer

### 1. PV на ярусе fp8: свернуть per-key скейл V в P

**Механизм.** Сегодня и в int8-, и в fp8-пути NInfer PV считается 16-битным MMA, потому что
V квантован по ключу и его скейл нельзя вынести из свёртки по ключам
(`ninfer/src/ops/softmax_attention/dense/causal_cache/small_t_i8.cuh:11-13`). Но выносить
его и не нужно: индекс, по которому меняется скейл V, — тот же самый `k`, по которому
индексируется P. Достаточно положить `P'[k] = P[k] · s_V[k]` **до** квантования P, и тогда
`acc[d] = Σ_k P'[k] · V_q[k][d]` — точное тождество, а не приближение. P' квантуется в
e4m3 с одним скейлом на строку тайла (максимум строки онлайн-софтмакс и так ведёт), V
читается из кеша как есть, и PV идёт `mma.kind::f8f6f4.m16n8k32`. Побочно исчезает
целиком проход декванта V в общую память вместе с его `__syncthreads`
(`prompt_i8.cuh:387-406`, `small_t_i8.cuh:539-558`) и bf16-тайл V в shared memory.

Формально из vLLM берётся только половина идеи: FA3 действительно квантует P на лету и
гоняет PV в fp8 (`flashinfer/include/flashinfer/attention/hopper/quantization/mainloop_mma.cuh:128`,
`kernel_traits.cuh:203-205`), а vLLM возит `prob_scale` и `bmm2_scale` как first-class
скейлы ровно под это (`vllm/v1/attention/backends/flashinfer.py:1918-1921`,
`vllm/model_executor/layers/quantization/kv_cache.py:160-190`). Но у них V-скейл
**на тензор**, поэтому он выносится тривиально; обобщение на per-key скейл — наше, в их
коде его нет. Это честно надо держать в голове: механизм подсказан их кодом, но не
скопирован из него.

**Ожидаемый эффект.** Внимание — 28–44% префилла (наш собственный замер), PV — примерно
половина FLOPs внимания. Ярус: сегодня PV на f16-операндах с f32-аккумулятором = 255.3
TFLOPS; с влитым патчем «PV fp16-acc» — 506.6. Целевой ярус `fp8 e4m3 f16acc` = 1028.9,
то есть **×2.03 к уже пропатченному состоянию и ×4.03 к текущему HEAD**. При доле PV
≈0.5·0.35 = 17.5% префилла ускорение вдвое даёт **≈8–9% префилла**; сверху — снятый проход
декванта V и один барьер на тайл. На декоде эффект меньше (декод упирается в HBM), но
освобождается shared memory под V-тайл, что может поднять занятость.

**Чем меряем.** Стенд 27B/35B-A3B, `--kv-dtype fp8`, `--prefill-chunk 8192`, mtp3.
A/B: HEAD с `mma_f16`-PV против ветки с fp8-PV, на длинах 1k/8k/32k, два прохода бенча,
свой worktree. Обязательно отдельно снять долю яруса: 32 MMA в теле цикла в SASS —
проверить, что ptxas действительно выдал `mma.kind::f8f6f4` с f16-аккумулятором, а не
откатился. Затем — точностная батарея A (AIME25), потому что это меняет вывод.

**Риски и что ломается.** (1) Вывод перестаёт быть побитово равным — мерить только на
стенде, где это допустимо. (2) e4m3 у P — 3 бита мантиссы, относительная ошибка ~6% на
элемент; по 64 ключам ошибки независимы, суммарная ~0.75% против ~0.05% у bf16-P.
Это на порядок хуже, и именно это надо проверять батареей, а не рассуждением. FA3 возит
это в продакшене, но с калиброванным `prob_scale`. (3) Работает только с **fp8**-кешем
для V; для int8-кеша `P·s_V` пришлось бы квантовать равномерно, а его динамический
диапазон задаётся разбросом `s_V` по ключам — там это сломается. Отсюда естественная
конфигурация: **K в int8-G64 (как сейчас, с Адамаром), V в fp8-row** — QK и PV всё равно
разные инструкции, смешивать форматы можно. (4) Регистровое давление: P' надо держать
в fp8-фрагментах нужной раскладки; возможен лишний `movmatrix`.

**Оценка объёма.** 4–6 файлов: `prompt_fp8.cuh`, `small_t_fp8.cuh`, `mma.cuh` (при
необходимости — вариант с f16-аккумулятором), кодек V, плюс тесты равенства.

### 2. Послойный формат KV (`kv_cache_dtype_skip_layers`)

**Механизм.** Разрешить задавать формат KV **по слою**, а не глобально: список индексов
слоёв (или классов слоёв), которые остаются в BF16 при глобальном `--kv-dtype int8|fp8`.
У vLLM это `--kv-cache-dtype-skip-layers` (`vllm/config/cache.py:114`), реализовано
одной веткой в конструкторе слоя внимания
(`vllm/model_executor/layers/attention/attention.py:290-306`: слой из списка получает
`kv_cache_dtype = "auto"`), а страницы выравниваются через `skip_page_size_padded`
(`vllm/platforms/interface.py:759-760`, `vllm/model_executor/layers/attention/attention.py:617-621`).
Для TurboQuant граничные слои подставляются автоматически
(`vllm/engine/arg_utils.py:2018-2026`), и там же сказано, зачем: без этого GSM8K на
Qwen3-4B падает на ~30 пунктов (`turboquant/config.py:190-192`).

**Ожидаемый эффект.** Не скорость, а **точность за известную цену по памяти**.
У 35B-A3B всего 10 слоёв с KV. Перевод одного слоя из int8 (1056 B/токен) в bf16
(2048 B/токен) даёт +9.4% к байтам KV на токен, двух — +18.8%. По декоду на 200k при
W ≈ 4 GB не-KV трафика: KV 2.11 → 2.31 GB, то есть **−3.2% декода** за два слоя.
Это дёшево, если батарея покажет, что первый/последний слой действительно несут
непропорционально много ошибки.

**Чем меряем.** Стенд B/C точностной батареи: `int8` целиком против `int8, кроме слоя 0`
против `int8, кроме слоёв 0 и последнего`. Сначала — дешёвая диагностика без прогонов
батареи: покомпонентная ошибка реконструкции K и V по слоям (L2 относительно bf16) на
десятке промптов; если разброс между слоями меньше 2×, кандидат закрывается сразу.

**Риски и что ломается.** Раскладка страниц: у нас страница жёстко 64 токена и профиль
`d256_profile.h:18-28` выбирает один тип на всё; понадобится либо отдельная арена для
bf16-слоёв, либо паддинг страниц до общего размера (у vLLM ровно эта проблема решается
`skip_page_size_padded`). Больше кода в аллокаторе, чем кажется. Побитовость вывода не
затрагивается — при пустом списке всё как сейчас.

**Оценка объёма.** 3–5 файлов: опции CLI/`EngineOptions`, `decoder_state.cpp`,
геометрия арены в `paged_kv_cache.cpp`, диспетчер ядер внимания.

### 3. Поиск скейла по ошибке реконструкции (`4over6`) — для NVFP4-**весов**, не для KV

**Механизм.** Вместо «скейл = absmax / max_repr» перебрать два (или несколько) кандидата
знаменателя и выбрать тот, у которого меньше L2-ошибка реконструкции блока. У vLLM это
`nvfp4_4over6`: кандидаты `max/6` и `max/4`, ошибка считается по уже загруженным в
регистры значениям с точной эмуляцией округления `e2m1`
(`vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:73-160`).

**Куда это у нас годится.** **Не в KV.** Для int8 на группе 64 после Адамара распределение
близко к гауссову; `E[max|x|]` по 64 отсчётам ≈ 2.75σ, а MSE-оптимальный порог клиппинга
для 8-битного равномерного квантователя гауссианы ≈ 3.9σ. То есть absmax уже **не
доклиппивает**, а значит уменьшать скейл (клиппинговать сильнее) заведомо хуже, а
увеличивать — некуда. При 255 уровнях выигрыш от выбора знаменателя пренебрежим — весь
смысл `4over6` в том, что у fp4 уровней шестнадцать.

А вот где это применимо — **реквантование NVFP4-весов** (у нас MLP-слои 0–55 у 27B, и
реквант GDN уже проходил гейт). Там группа 16, уровней 16, и выбор `max/6` против `max/4`
меняет ошибку заметно. Цена — нулевая на инференсе: поиск делается один раз при
реквантовании.

**Чем меряем.** Оффлайн: L2-ошибка реконструкции весов по слоям, absmax/6 против
min-MSE; затем точностная батарея A на реквантованном чекпоинте. Скорость не трогается
вообще, поэтому A/B по времени не нужен.

**Риски.** Меняются веса → меняется вывод. Никакого влияния на производительность,
никакого влияния на раскладку.

**Оценка объёма.** 1–2 файла в утилите реквантования. Это чужая для этой ячейки область;
передать в ячейку про NVFP4.

---

## Опровергнуто / не переносится

- **«FP8-кеш даёт другой, более быстрый маршрут ядра».** На sm_120 fp8-MMA во внимании
  не задействован нигде: `prefill.cuh` всегда `m16n16k16 f16f16f32`
  (`flashinfer/.../prefill.cuh:1289-1293`, `:1823-1827`), а готовый
  `mma_sync_m16n16k32_row_col_f8f8f32` (`flashinfer/include/flashinfer/mma.cuh:217`)
  **не вызывается ниоткуда**. Настоящий fp8-путь — только FA3/sm90 (`wgmma`, которого
  на нашей карте нет по брифу) и trtllm-gen/sm100. Заимствовать нечего: у нас QK уже
  идёт на `mma.kind::f8f6f4` (`ninfer/src/ops/common/mma.cuh:59-66`), то есть **мы в этом
  месте впереди FlashInfer на этой карте**.
- **Гранулярность скейлов.** Их обычный fp8 — один скаляр на весь тензор, из чекпоинта,
  по умолчанию 1.0 (`vllm/model_executor/layers/quantization/kv_cache.py:105-108,126-129`).
  Наш int8 — absmax по группе 64 каналов, динамически, плюс Адамар
  (`ninfer/src/ops/kv_cache/int8_g64_codec.cuh:19-21`,
  `ninfer/src/ops/kv_cache/hadamard_d256.cuh:46-65`). Их режим `per_token_head` — один
  скейл на (токен, голову), то есть **в 4 раза грубее нашего**
  (`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:207`). Перенимать нечего.
- **Скейлы внутрь страницы (inline padding).** У них
  `[K(hs) | K_scale(4B) | V(hs) | V_scale(4B)]` на слот
  (`vllm/v1/attention/backends/triton_attn.py:351-369`). У нас скейлы — отдельная плоскость
  `FP16 [4, 64, kv_heads, pages]` (`ninfer/src/targets/qwen3_6/impl/state/decoder_state.cpp:44-46`).
  Считаем: на (страницу, голову) коды занимают 64·256 = 16 384 B, скейлы 64·4·2 = 512 B,
  то есть **3.1% трафика, уже одним непрерывным куском в 512 B**. Инлайн не убирает ни
  одной транзакции — он только ломает 256-байтное выравнивание строк кодов и мешает
  векторному `cp.async` по 16 B. Отклонено арифметикой.
- **TurboQuant целиком.** Основной CUDA-путь **не использует тензорные ядра вообще**:
  деквант в fp32-регистры и `tl.sum` (`vllm/v1/attention/ops/triton_turboquant_decode.py:208-211`,
  `:304`, `num_warps=1` на `:586`). Кодовая книга Ллойда—Макса неравномерна и по построению
  **не может кормить `mma.s8s8s32`** — а у нас QK на нём и работает. Плюс их Адамар и
  вращение Q мы уже имеем (`ninfer/src/ops/kv_cache/hadamard_d256.cuh`, `prompt_i8.cuh:156`).
  Остаётся только идея нормировать K на L2-норму вместо absmax — при 8 битах это не
  выигрыш. Соседний движок уже реализовал TurboQuant и снял его в пользу MXFP4; наш разбор
  это подтверждает независимо.
- **`mock kv cache` — как делать не надо.** На SM100 vLLM перед trtllm-префиллом
  **материализует целиком bf16-копию всего окна KV этого префилла** отдельным Triton-ядром
  (`vllm/v1/attention/backends/flashinfer.py:227-278`, `torch.empty(new_s, dtype=bf16)`
  на `:249`, вызов на `:2238`). Это ровно тот антипаттерн, который у нас уже был измерен и
  отвергнут как «route B» (−37% на префилле из-за конверсии V каждой CTA). Не переносить.
- **Гипотеза (а), планировщик.** Опровергнута выше построчно.
- **Гипотеза (в), перенос места квантования.** Опровергнута: место одно и то же, fp8 только
  добавляет операции.
- **Ширина чанка как объяснение fp8-эффекта.** Опровергнута: `cache_dtype` не входит ни
  в одно выражение `SchedulerConfig`. При этом строка `--max-num-batched-tokens 8192`
  из той же таблицы — настоящая и уже наша: у vLLM дефолт API-сервера 2048
  (`vllm/engine/arg_utils.py:2602-2611`), у нас в коде дефолт 1024
  (`ninfer/include/ninfer/types.h:116`) и рабочие 8192 задаются флагом.
- **NVFP4-KV.** Раскладка со swizzle под trtllm-gen привязана к SM100
  (`vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:33-47`), а на нашей карте
  блок-скейл MMA принимает только f32-аккумулятор (бриф). Для внимания это означает fp4
  и у Q тоже; при head_dim 256 и уже существующем точном int8-ярусе не окупается.
  Из всего файла берём только поиск скейла по MSE, и то для весов (кандидат 3).

---

## Открытые вопросы

1. **Ёмкость 286k → 462k = 1.62×, а механизм предсказывает ровно 2.0×.** Страница внимания
   ровно вдвое меньше по байтам (`vllm/v1/kv_cache_interface.py:243-259`), страница GDN
   добивается до неё точно (`vllm/platforms/interface.py:920-940`), число блоков —
   `available_memory // page_size // num_layers` (`vllm/v1/core/kv_cache_utils.py:988`).
   Все три шага дают чистую двойку. Откуда 1.62 — из кода не выводится. Плюс сама
   постановка задачи внутренне противоречива: в таблице 286k, в тексте «с 254k до 462k».
   Нужны обе строки из лога `GPU KV cache size` вместе с `block_size`.
2. **Какой бэкенд был выбран фактически.** Весь разбор гипотезы (б) держится на том, что
   в прогоне по умолчанию работал `FLASH_ATTN`, а с fp8 — `FLASHINFER`. Это надо просто
   прочитать в логе; если оператор в обоих прогонах явно задавал `VLLM_ATTENTION_BACKEND`,
   объяснение рассыпается и загадку надо открывать заново.
3. **Доступен ли был XQA.** `vllm/utils/flashinfer.py:407-408` — XQA-кубины качаются с
   артифактори NVIDIA; без сети декод откатывается на FA2 с `CTA_TILE_Q=16`. Это меняет
   интерпретацию декодной колонки (хотя байтовая модель её и так объясняет с одним
   параметром).
4. **head_dim 256 и `CTA_TILE_Q`.** `FA2DetermineCtaTileQ`
   (`flashinfer/include/flashinfer/utils.cuh:408-438`) даёт 128 только при `head_dim < 256`.
   Что именно выбирается при 256 и как это влияет на цену распаковки — не прослежено.
5. **Калибровка `prob_scale`.** vLLM предупреждает про «uncalibrated q_scale/prob_scale»
   (`vllm/model_executor/layers/quantization/kv_cache.py:184-190`), но чем их калибруют —
   в этом дереве нет. Для кандидата 1 это ключевой вопрос: нужен ли статический скейл P
   вообще, если онлайн-софтмакс и так даёт максимум строки.
6. **Патч «PV fp16-acc».** В клоне `da49c0d` его нет, а журнал экспериментов на него
   ссылается (+4.3..9.2%). Арифметика кандидата 1 приведена относительно обоих состояний,
   но базу надо зафиксировать перед замером.
