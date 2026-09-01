# Q1 — формат чекпоинта решает, какое ядро включится

## Что это за механизм

Семь сборок различаются не «качеством кванта», а **строкой в метаданных**, по которой движок
выбирает ядро. У compressed-tensors решение принимается по наличию блока `input_activations` в
`config_groups`: если он есть — включается нативный FP4-путь (cutlass/flashinfer), если его нет —
принудительно Marlin с распаковкой в 16 бит. У ModelOpt то же решение записано буквальной строкой
`quant_algo` на слой: `"NVFP4"` против `"W4A16_NVFP4"`. Оба ядра поддержаны на sm_120 (Marlin с
SM75, cutlass FP4 с SM120), то есть **железо ничего не выбирает — выбирает файл**. Отчёт по этому
пункту прав; ниже дословные условия из четырёх независимых реализаций.

Второй слой механизма — сам формат: NVFP4 это E2M1-коды + **FP8-E4M3 скейл на 16 значений** +
один FP32 глобальный делитель на тензор; AWQ/GPTQ/AutoRound — это int4 с **асимметричным** нулём
и FP16-скейлом на группу 32/128. Разные алгоритмы (RTN / GPTQ / AWQ / AutoRound / AutoQuantize)
производят **одни и те же байты** в этих форматах — отличается только процедура выбора кодов.
Именно поэтому направление «лучший энкодер при том же формате» для нас открыто без единого
изменения в ядрах.

## Доказательства из кода

### Диспатч ядра: что именно решает

- `vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:404-421` —
  `_is_nvfp4_format()`: `strategy == TENSOR_GROUP && type == FLOAT && num_bits == 4 &&
  group_size == 16 && symmetric`. Группа **ровно 16**, иначе это не NVFP4.
- То же место `:734-743` — вся развилка:
  ```python
  if self._is_nvfp4_format(weight_quant):
      if input_quant is None:
          return CompressedTensorsW4A4Fp4(use_a16=True)
      if not self._is_nvfp4_format(input_quant):
          raise ValueError(...)
      return CompressedTensorsW4A4Fp4()
  ```
  Единственный различитель — `input_quant is None`.
- `vllm/vllm/model_executor/kernels/linear/__init__.py:1047-1049` — куда ведёт флаг:
  ```python
  elif linear_backend == "auto" and use_a16:
      # Force a16 (Marlin) when running weight-only quantization.
      force_kernel = MarlinNvFp4LinearKernel
  ```
  Это не фолбэк по неподдержке, а безусловный форс.
- `vllm/vllm/model_executor/kernels/linear/nvfp4/marlin.py:21-27` — Marlin доступен, если
  `is_fp4_marlin_supported()`; `vllm/.../quantization/utils/marlin_utils_fp4.py:34-35` —
  это просто `current_platform.has_device_capability(75)`.
- `vllm/vllm/model_executor/kernels/linear/nvfp4/cutlass.py:23-29` → `.../utils/nvfp4_utils.py:56-61`
  → `vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu:82-84`:
  `#if ENABLE_NVFP4_SM120 ... if (cc >= 120 && cc < 130) return true;` — **на 5090 cutlass FP4
  собран и доступен**. Значит выбор Marlin у `nvidia/…` вызван метаданными, а не железом.
- `vllm/vllm/model_executor/kernels/linear/__init__.py:525-542` — порядок приоритета
  `_POSSIBLE_NVFP4_KERNELS`: FlashInfer(CuteDSL, Cutlass, B12x) → `CutlassNvFp4LinearKernel` →
  `MarlinNvFp4LinearKernel` → … Marlin шестой, до него доходят только W4A16.

### То же самое на MoE (а у 35B-A3B вес именно там)

- `vllm/.../compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:152` —
  `CompressedTensorsW4A4Nvfp4MoEMethod(..., use_a16=(input_quant is None))`.
- `vllm/.../compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:60` (и `modelopt.py:1395`) —
  `activation_key=None if use_a16 else kNvfp4Dynamic`.
- `vllm/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py:718-724` —
  `_supports_quant_scheme(...) -> return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)`.
  При `activation_key=None` cutlass-эксперты **отвечают False**, перебор в
  `vllm/.../fused_moe/oracle/nvfp4.py:180-188` доходит до `NvFp4MoeBackend.MARLIN`.
  Это второй, независимый путь к тому же результату.

### ModelOpt MIXED_PRECISION: правило записано в файле как строка

- `vllm/vllm/model_executor/layers/quantization/modelopt.py:2143-2151` — конфиг читает
  `quantized_layers` из `quantization_config` (или legacy `hf_quant_config.json`).
- `modelopt.py:2388-2400` — диспатч по строке:
  `"FP8"` → `ModelOptFp8LinearMethod`; `"NVFP4"` → `ModelOptNvFp4LinearMethod` (W4A4);
  `"W4A16_NVFP4"` → `ModelOptNvFp4W4A16LinearMethod`; иначе `UnquantizedLinearMethod()`.
- `modelopt.py:1248-1255` — в W4A16-методе `init_nvfp4_linear_kernel(use_a16=True)` с комментарием
  «`use_a16=True` forces Marlin … avoiding a W4A4 kernel that requires input_scale».
- `modelopt.py:1234, 1299-1304` — формат на диске: `weight` uint8 (2 нибла/байт по входной оси),
  `weight_scale` fp8-e4m3 на 16 элементов, `weight_scale_2` fp32 = **amax / (6.0 × 448.0) = amax/2688**.
  ModelOpt хранит делитель, compressed-tensors — обратную величину (`:110-114` в схеме CT).
- `modelopt.py:2354-2370` — `_quantized_layer_prefix_candidates` явно обрабатывает `lm_head`:
  голова в этой схеме — обычная запись в `quantized_layers`, ничем не выделенная.

### То же в SGLang и в третьем движке (независимое подтверждение)

- `sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py:516-542` —
  `_is_fp4a4_nvfp4` требует TENSOR_GROUP **и у весов, и у активаций**, group_size 16 у обоих;
  при `input_quant is None` возвращает False на первой строке.
- Там же `:647-654` — cutlass-путь только внутри `is_activation_quantization_format(quant_format)`;
  весовой-только чекпоинт уходит в `_is_wNa16_group_channel` (`:544-557`) → `CompressedTensorsWNA16` → Marlin.
- `sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:729, 815-818, 955, 982` — та же
  таблица `quant_algo` с `"W4A16_NVFP4"`.
- `tokenspeed/python/tokenspeed/runtime/layers/quantization/modelopt_mixed.py:189-192, 205, 322-335` —
  третья независимая реализация того же правила.

### Что Marlin делает со скейлами (важно для памяти)

- `vllm/.../utils/marlin_utils_fp4.py:266-287` — при загрузке `weight_scale` (fp8 на диске)
  приводится `weight_scale.to(param_dtype)` (bf16/fp16), переставляется и уходит в
  `nvfp4_marlin_process_scales` (формат S0E5M3, `:61-82`). **В VRAM скейл-плоскость вдвое
  толще, чем на диске.** Итого Marlin-путь: 4 + 16/16 = **5.0 бит/вес** против 4.5 бит у
  cutlass-пути на тех же данных → **+11.1 % памяти под веса**.
- `vllm/.../quantization/utils/marlin_utils.py:36` — `MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]`.
  Для промышленного W4A16-ядра группа 64 — штатный размер, не экзотика.

### Формат файла: что где лежит

- `comprtensors/src/compressed_tensors/config/base.py:15-27` — перечень форматов:
  `pack-quantized`, `nvfp4-pack-quantized`, `mxfp4-pack-quantized`, `mixed-precision`, …
- `comprtensors/src/compressed_tensors/compressors/nvfp4/base.py:35-45` — набор тензоров NVFP4:
  `weight_packed`, `weight_scale`, `weight_global_scale` (+ `input_global_scale`, если активации
  статические). `:128-138` — `can_compress` требует **group_size == 16**.
- `comprtensors/src/compressed_tensors/compressors/nvfp4/helpers.py:20-22, 87-108` — упаковка
  E2M1: индексы `(0, 0.5, 1, 1.5, 2, 3, 4, 6)`, два нибла в байт, младший — первый по K.
- `comprtensors/src/compressed_tensors/quantization/utils/helpers.py:320-349` — `generate_gparam`:
  `global_scale = 448 × 6 / amax_tensor`.
- Там же `:80-109` — `calculate_qparams`: групповой скейл =
  `round_to_fp8_e4m3(global_scale × amax_group / 6)`. Двухуровневая схема, скейл занимает весь
  диапазон E4M3.
- `comprtensors/src/compressed_tensors/quantization/quant_scheme.py:166-203` — пресеты
  **NVFP4A16** (только `weights`) и **NVFP4** (`weights` + `input_activations` с
  `dynamic=LOCAL`, `observer="static_minmax"`). Это ровно те два варианта, что разводят Marlin и cutlass.
- `comprtensors/.../quant_scheme.py:204-238` — MXFP4: `strategy=GROUP`, `group_size=32`,
  `scale_dtype=uint8` (E8M0, степень двойки). Другой класс точности скейла.
- `vllm/.../quantization/auto_awq.py:74-76` — AWQ пакует нибблы в int32 в порядке
  `[0,4,1,5,2,6,3,7]` по **выходной** оси; `:492-494` — тензоры `qweight`, `qzeros`, `scales`
  (асимметричный, есть нули).
- `vllm/.../quantization/auto_gptq.py:442-445` — GPTQ: `qweight`, `g_idx` (перестановка act-order),
  `scales`, `qzeros`. Тоже асимметричный.
- `sglang/.../quantization/auto_round.py:50-52` — AutoRound **не имеет своего формата**:
  `SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}`. Это алгоритм поверх
  чужой упаковки. `:190-236` — `extra_config` даёт **побитовую разбивку по именам слоёв**
  (`{bits, group_size, sym}` на слой) — это и есть механизм «int4-mixed».

### Алгоритмы: что есть в клоне

- `marlin/gptq/gptq.py:23-56` — GPTQ строит `H = (2/n)·Σ x xᵀ` по калибровочным активациям.
- `marlin/gptq/gptq.py:55-144` — `fasterquant`: демпфирование `percdamp·mean(diag H)`, Cholesky
  инверсия, поколоночное квантование с **подстановкой ошибки в оставшиеся колонки**
  `W1[:, i:] -= err1 ⊗ Hinv1[i, i:]`, блоками по 128.
- `marlin/gptq/quant.py:52-70` — поиск клипа: сетка `grid=100`, `maxshrink=0.75` (76 кандидатов),
  критерий — гессиан-взвешенная ошибка `‖solve_triangular(Hinv, δ)‖²`. Это ровно тот «поиск скейла»,
  которого у нас нет.
- `marlin/gptq/llama2.py:243-262` — дефолты: `nsamples=256`, `seqlen=2048` (≈524 288 токенов),
  `percdamp=0.1`, `groupsize=128`.
- `comprtensors/src/compressed_tensors/transform/transform_args.py:12-44` — таблица локаций
  поворотов: `WEIGHT_INPUT` / `WEIGHT_OUTPUT` помечены **offline**, `INPUT`/`OUTPUT`/`K_CACHE`/`Q_ATTN` — online.
- `comprtensors/.../transform/factory/base.py:127-137` — offline-поворот **вплавляется в веса**
  (`update_offload_parameter(module, "weight", transform(module.weight))`), и трансформ удаляется:
  «transform is no longer needed». `factory/hadamard.py:44` — offline-повороты считаются в **float64**.
  То есть R-повороты Адамара при `WEIGHT_INPUT/OUTPUT` — чисто конвертерная операция, ноль во время исполнения.
- `trtllm/examples/quantization/quantize_mixed_precision_moe.py:281-305` — эталонное
  правило смешанной точности от NVIDIA (DeepSeek): **написано руками**, не найдено алгоритмом:
  attention (`fused_a`, `q_b_proj`, `kv_b_proj`, `o_proj`), shared experts и **MLP первых трёх
  слоёв** → `FP8_BLOCK_SCALES`; routed experts слоёв 3..60 → `W4A8_AWQ`.
- Там же `:129-148` — как на самом деле получены «AWQ»-веса: `abs().max(dim=2)/7`,
  `clamp(round(w/s), -8, 7)` — **обычный RTN, group 128**. «AWQ» здесь только про
  активационные скейлы из калибровочных amax (`:111-131`, `amax/448`).

### Что у нас сегодня

- **Энкодер весов — чистый RTN, без калибровки.**
  `ninfer/tools/convert/common/quantize.py:96-112`: `max_abs = grouped.abs().amax(dim=2)`,
  скейл = `max_abs/qmax` округлённый до binary16, коды = `clamp(round(x · 1/scale), qmin, qmax)`.
  Ни гессиана, ни подстановки ошибки, ни поиска клипа, ни поворотов.
- **Реестр форматов** `ninfer/tools/artifact/numeric.py:53-58`:
  `Q4G64_F16S(4, 64, -8, 7)`, `Q5G64_F16S(5, 64)`, `Q6G64_F16S(6, 64)`, `W8G32_F16S(8, 32, -127, 127)`,
  `NVFP4(group 16)`, `FP8_E4M3FN_ROW_BF16S`. Все int-форматы **симметричные, без нуля** —
  чекпоинт GPTQ/AWQ (асимметричный) импортировать «как есть» нельзя, переносится только алгоритм.
- **Раскладка** `ninfer/tools/artifact/layouts.py:33, 182-220` — `row-split-k128-v1`: K выравнивается
  до 128, `groups_per_row = k_pad / group_size`, 2 байта скейла на группу, Q5/Q6 = базовая
  4-битная плоскость + высокая плоскость. Раскладка **параметрична по group_size**.
- **Ядро — нет.** `ninfer/src/ops/linear/q4/q4_rowsplit_storage.cuh:12-16`:
  `kGroupK = 64`, `kCodeBytesPerGroup = 32` — константы компиляции. То же в
  `q5_rowsplit_storage.cuh:13`, `q5_rowsplit_gemv.cuh:60,112`, `q6_rowsplit_storage.cuh:13`.
- **Q4/Q5/Q6 считаются на bf16-ярусе.**
  `ninfer/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:264` — ядро
  `sparse_moe_prefill_q4_gate_up_kernel`; `:355` — `Q4MmaDecodeAtom::decode_eight`, который по
  `ninfer/src/ops/linear/q4/q4_rowsplit_storage.cuh:48-60` собирает **bf16-слова**
  (`0x43084308` = bf16 136.0, `0x43004300` = bf16 128.0); `:401` — результат уходит в `mma_bf16`
  (`ninfer/src/ops/common/mma.cuh:32-40` = 255.3 TFLOPS). Q5/Q6 идут тем же путём
  (`:660-679`, `:776`). Наши 4 бита — это формат **трафика**, а не вычислений; группа для них
  ничем в железе не зафиксирована.
- **NVFP4 — единственный формат с аппаратным ограничением группы.**
  `ninfer/src/ops/common/mma.cuh:84-103`: `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` —
  4 скейла на k=64, то есть **ровно один скейл на 16 значений**. Группа 32 означала бы `mxf4`
  с ue8m0-скейлом (степень двойки) — строго хуже численно.
- **Смешанная точность 35B: ручной список из трёх слоёв.**
  `ninfer/tools/convert/qwen3_6_35b_a3b/inventory.py:37,42` — `Q6_ROUTED_DOWN_LAYERS = (34, 38, 39)`,
  остальные 37 слоёв `routed_down` в Q5. `ninfer/docs/maintainer/qwen3.6-35b-a3b-artifact.md:250`
  фиксирует список, **но не приводит критерия выбора**. Метрики чувствительности в дереве нет
  (`grep sensitiv|kl_div` по `docs/`, `tools/` — только не относящиеся к делу совпадения).
- **Смешанная точность 27B NVFP4 — не наша.**
  `ninfer/tools/convert/qwen3_8_27b/inventory_nvfp4.py:46-47` — `NVFP4_MLP_LAYERS = range(56)`,
  `FP8_MLP_LAYERS = range(56, 64)`. Источник: `ninfer/tools/convert/qwen3_8_27b/recipe_nvfp4.py:19-23`
  — `BASE_REPOSITORY = "Qwen/Qwen3.8-27B"`, `QUANTIZED_REPOSITORY = "unsloth/Qwen3.8-27B-NVFP4"`.
  `recipe_nvfp4.py:662-707` и `618-664` — и NVFP4-, и FP8-веса **копируются из unsloth**
  (`weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`) с проверкой сигнатуры,
  без переквантования. То есть правило «MLP 0-55 в NVFP4, остальное FP8» — это **рецепт unsloth
  (llm-compressor)**, а не наш выбор.
- Это зафиксировано и в документации: `ninfer/docs/maintainer/tensor-formats.md:112-117` —
  «`NVFP4` … has **no NInfer-owned canonical source-to-NVFP4 encoder**. Its current checkpoint recipe
  copies already selected E2M1, E4M3FN, and FP32 divisor words from its fixed source».
  И `:107-112` — прямое разрешение подменить энкодер: «A checkpoint-specific process may use another
  documented encoder to produce one of those schemes — for example, an **upstream error-optimized
  recipe** — but that does not create a new quantization scheme».
- **Голову мы уже квантуем.** 35B: `ninfer/tools/convert/qwen3_6_35b_a3b/inventory.py:98` —
  `text/output_head [248320,2048]` в **Q6G64_F16S**; `:47` — `token_embedding` в W8; `:106` —
  `draft_head [131072,2048]` в Q4. 27B NVFP4:
  `ninfer/tools/convert/qwen3_8_27b/inventory_nvfp4.py:70,147` — `token_embedding` и `output_head`
  в **FP8 row-scale**, энкодер свой: `ninfer/tools/convert/qwen3_8_27b/fp8_embedding.py:21`
  — профиль `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1` (снова absmax + RNE, без калибровки).
- **W4A16 для NVFP4 у нас уже есть.** `ninfer/src/ops/linear/nvfp4/nvfp4_dispatch.cpp:20-42` —
  `Nvfp4LinearRoute::{A16, W4A4}`; `LinearPolicy::A16Only` даёт A16 безусловно, иначе W4A4
  включается только при `tokens >= 4/5/8` по классу задачи. То есть в декоде мы и так идём
  весовым-только путём; активационный делитель нужен только префиллу.
- **Но список задач закрыт.** `ninfer/src/ops/linear/nvfp4/nvfp4_config.h:118-124` —
  `enum class Nvfp4Problem {AttnInput, GdnInput, MlpGateUp, Residual6144, Residual17408}` и
  `:126-160` — белый список по точным `(N, K)`. Форма головы туда не входит.
  Геометрическое ограничение при этом выполняется:
  `ninfer/src/ops/linear/nvfp4/nvfp4_format.cpp:38-41` требует `N % 128 == 0 && K % 64 == 0`;
  `248320 = 128·1940`, `5120 = 64·80`, `2048 = 64·32`.
- **Наши собственные числа точности** (одна выборка, EvalScope 1.9.0, MTP=3, INT8 g64 KV):
  `ninfer/model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md:298-305` — NVFP4-сборка (энкодер unsloth):
  GPQA-Diamond **90.40 %**, AIME25/26 96.67 %, IFBench 77.00, RealWorldQA 83.53.
  `ninfer/model-cards/Qwen3.8-27B-NInfer/README.md:214-220` — наша int-сборка (наш RTN):
  GPQA-Diamond **87.37 %**, AIME25/26 96.67 %, RealWorldQA 82.22. Официальный BF16 — GPQA 89.2.
  Разрыв 3.03 п.п. на 198 вопросах при σ_бином ≈ 2.1 п.п. — **сигнал, но не доказательство**;
  честно это «наш RTN не показал превосходства над чужим энкодером».

## Кандидаты для NInfer

### 1. GPTQ-энкодер для существующих форматов Q4/Q5/Q6/W8 (самый крупный)

**Механизм.** Заменить `quantize_matrix` (`tools/convert/common/quantize.py:96-112`) на
послойный GPTQ: собрать `H = (2/n)Σxxᵀ` на калибровочном корпусе, демпфировать, инвертировать
через Cholesky и квантовать колонки по одной, подставляя ошибку в остаток
(`marlin/gptq/gptq.py:88-120`), плюс поиск клипа по гессиан-взвешенной метрике
(`marlin/gptq/quant.py:52-70`). Выходные байты — те же `row-split-k128-v1` коды и binary16 скейлы,
симметричные, без нуля; декодер, ядра и побитовость исполнения не меняются вообще.
Документация это прямо разрешает (`docs/maintainer/tensor-formats.md:107-112`).

**Ожидаемый эффект.** Скорости не меняет — **ноль байт, ноль FLOP**. Меняет только точность.
Ориентир из веба: AutoRound Table 5 (Mistral-7B, W4 per-channel) RTN 58.84 → +клип 61.10 →
+округление 61.62 → оба 62.33 при FP16 63.30 (https://arxiv.org/abs/2309.05516). На группе 64
разрыв меньше, чем на per-channel, но именно наш самый «горячий» тензор — `routed_gate_up` в Q4 —
сидит на 4 битах, где отдача максимальна. Дополнительный ориентир: palmfuture опубликовал для
GPTQ-сборки того же 35B-A3B `wikitext-2` PPL 6.1846 ≈ 97.9 % от BF16
(https://huggingface.co/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4).

**Чем меряем.** `ninfer-perplexity` (`docs/perplexity.md`) на фиксированном корпусе
`eval/corpora/perplexity-1m` — A/B двух артефактов, одинаковых по формату и раскладке,
различающихся только энкодером. Затем батарея AIME25 / GPQA-diamond / LongBench на трёх руках,
куда добавляется рука `Q1_gptq`. Порядок: сперва PPL (дешёвая, чувствительная), батарея — только
если PPL сдвинулась.

**Риски и что ломается.** Вывод перестаёт быть побитово равным текущему артефакту — но это новый
артефакт, а не изменение существующего; текущий остаётся валидным. Нужен калибровочный корпус и
один прогон forward на GPU (сервер занят батареей — планировать после). Гессиан на
`routed_gate_up [262144, 2048]` считается по эксперту (K=2048 → H 2048², 16 МБ fp32) —
это дёшево; узкое место в том, что каждый из 256 экспертов на 40 слоях видит только свою долю
токенов, и для редких экспертов H будет плохо обусловлен → нужен фолбэк на RTN при малом
`nsamples` (у palmfuture ровно так: 97.42 % модулей GPTQ, 2.58 % RTN-фолбэк).

**Оценка объёма.** 1 новый файл в `tools/convert/common/` (~250 строк), правка выбора энкодера
в 3-4 файлах рецептов, ноль файлов в `src/`.

### 2. Офлайн-повороты Адамара, вплавленные в веса

**Механизм.** Пара `R` на выходе матрицы и `Rᵀ` на входе следующей — тождество в точной
арифметике, но выравнивает выбросы по каналам и заметно снижает ошибку 4-битного кванта обеих.
compressed-tensors делает это в конвертере (`transform/factory/base.py:127-137`), считая поворот
в float64 (`factory/hadamard.py:44`) и **удаляя трансформ после вплавления** — во время
исполнения ничего не остаётся. У нас Адамар уже реализован для K/Q
(`src/ops/kv_cache/hadamard_d256.cuh`), то есть математика знакома.

**Ожидаемый эффект.** Ноль байт и ноль FLOP на рантайме; эффект целиком в точности, и он
складывается с кандидатом 1 (повороты и GPTQ ортогональны). Основная мишень — та же
`routed_gate_up` в Q4.

**Чем меряем.** Тот же `ninfer-perplexity` A/B. Сначала — численная проверка тождества:
максимальное относительное отклонение выхода слоя bf16-модели с поворотом и без него должно
быть на уровне round-off.

**Риски и что ломается.** Пара «вплавляется» только там, где между матрицами нет поканальной
нелинейности. У нас между `o_proj` и следующим `post_attention_norm` стоит RMSNorm с
поканальным весом — его нужно сначала абсорбировать в следующую матрицу, иначе поворот не
тождественен. В MoE вход `routed_gate_up` общий для 256 экспертов — поворот входа применим
(один `R` на слой), а вот `Rᵀ` надо посадить на выход предыдущей `down`, что через маршрутизацию
и веса экспертов не проходит тривиально. **Это главный риск: возможно, применимых пар в нашем
графе окажется мало.** Инвентаризацию пар надо сделать до кода.

**Оценка объёма.** ~150 строк в конвертере + инвентаризация пар; ноль файлов в `src/`.

### 3. Голова 27B NVFP4: FP8 row → NVFP4 (единственный кандидат с прямой арифметикой скорости)

**Механизм.** `text/output_head [248320, 5120]` сейчас FP8 E4M3 с bf16-скейлом на строку
(`inventory_nvfp4.py:147`). Перевести в NVFP4 и гнать через уже существующий A16-маршрут
(`nvfp4_dispatch.cpp:25`), без квантования активаций — логиты остаются функцией bf16-входа.

**Ожидаемый эффект (арифметика).**
FP8: коды 248320·5120 = 1 271 398 400 Б = 1212.5 МиБ, скейлы 248320·2 Б = 0.47 МиБ → **1.272 ГБ**.
NVFP4: коды 606.25 МиБ + скейлы 248320·320·1 Б = 75.8 МиБ → **0.715 ГБ**. Экономия **0.557 ГБ**
на один полный проход головы. При декоде 27B ≈ 180 ток/с это 100.2 ГБ/с = **5.6 % от пика HBM
1792 ГБ/с**. Верхняя граница выигрыша в декоде — те же 5.6 %, если декод полностью упирается в HBM.

Для 35B голова уже в Q6G64 (363.4 МиБ кодов + 15.2 МиБ скейлов = 0.397 ГБ); переход в NVFP4 дал бы
0.286 ГБ, экономия 0.111 ГБ → при 579 ток/с это 64.3 ГБ/с = **3.6 % от пика**. Это существенно
меньше «+10.4 % декода», которые дал соседний движок, — потому что у них голова была толще нашей.
**Основной выигрыш на голове мы уже взяли; остаток ≈ 3.6 % (35B) и ≈ 5.6 % (27B).**

**Чем меряем.** Декод-стенд, канонический замер (чанк 8192 + mtp3), A/B двух артефактов,
отличающихся только форматом `text/output_head`. Точность — `ninfer-perplexity` + IFBench
(веб-источники единодушно указывают, что квантование головы бьёт именно по инструкциям:
llm-compressor issue #1820, «accuracy is very sensitive to its quantization»;
AutoRound отмечает «significant accuracy drop on IFEVAL for Qwen2.5-14B-Instruct»,
https://medium.com/intel-analytics-software/10-tips-for-quantizing-llms-and-vlms-with-autoround-923e733879a7).

**Риски и что ломается.** (а) Нужен **собственный source→NVFP4 энкодер** — его в дереве нет по
`docs/maintainer/tensor-formats.md:112-117`; это самая большая часть работы.
(б) `Nvfp4Problem` — закрытый белый список (`nvfp4_config.h:118-124`), нужна новая геометрия и
подбор расписания. (в) Голова — самое чувствительное место по единодушному мнению всех трёх
вендоров (NVIDIA, RedHat, Intel); NVIDIA в претрейн-статье оставляет «embeddings, the output
projection head» в исходной точности всегда (arXiv:2509.25149). У нас, впрочем, голова уже
квантована и GPQA не просела — значит запрет не абсолютный, а про 4 бита конкретно.

**Оценка объёма.** ~300 строк энкодера в конвертере + 1 геометрия и запись в enum в
`src/ops/linear/nvfp4/` (2-3 файла) + инвентарь/рецепт.

### 4. Группа 128 вместо 64 для routed-экспертов (лёгкий выигрыш байт, если точность позволит)

**Механизм.** Добавить в реестр `Q4G128_F16S` / `Q5G128_F16S`. Раскладка `row-split-k128-v1`
уже параметрична (`layouts.py:190`), в ядрах меняется одна константа `kGroupK`.

**Ожидаемый эффект (арифметика, 35B-A3B, на слой).**
`routed_gate_up [262144,2048]` Q4G64: коды 256 МиБ, скейлы 262144·32·2 = **16 МиБ (6.25 %)**.
`routed_down [524288,512]` Q5G64: коды 160 МиБ, скейлы 524288·8·2 = **8 МиБ (5.0 %)**.
На слой 440 МиБ, из них 24 МиБ скейлов (5.45 %). На 40 слоёв — 17.19 ГиБ, из них 0.9375 ГиБ скейлов.
Переход на 128 убирает **480 МиБ = 2.73 % байт routed-экспертов**. При декоде, идущем близко к
потолку HBM, это верхняя граница ускорения ≈ **2.7 %** и столько же освобождённого VRAM
(≈ +2.7 % контекста).
Обратное движение (64 → 32) стоит +960 МиБ (+5.45 %) и покупает точность.

**Чем меряем.** Двухсторонний A/B: три артефакта g32 / g64 / g128, одинаковый энкодер;
скорость на каноническом декод-стенде, точность — `ninfer-perplexity`. Именно этот замер
закрывает пункт (в) задания, потому что **публичных аблаций 16/32/64/128 на 4 битах не
существует** (см. «Опровергнуто»).

**Риски и что ломается.** `kGroupK` — константа компиляции в 6+ заголовках
(`q4_rowsplit_storage.cuh:13`, `q5_rowsplit_storage.cuh:13`, `q5_rowsplit_gemv.cuh:60,112`,
`q6_rowsplit_storage.cuh:13`, `q4_rowsplit_gemm_mma.cuh` swizzle на 64). Свизл и расписание
стадий завязаны на 64 значения на группу — при 128 меняется и разбиение на стадии.
Занятость и давление на регистры надо перепроверять. Ожидаемая цена в точности — по закону
Dettmers&Zettlemoyer (ICML 2023) переход 1024→64 стоит 0.24 бита и даёт почти столько же,
сколько +1 бит; их рекомендация — «block size **128 or lower**», то есть 64→128 может оказаться
почти бесплатным, а 64→32 — почти бесполезным.

**Оценка объёма.** 2 строки в реестре форматов, ~6 заголовков ядер, перетюн расписаний.

### 5. Правило смешанной точности: заменить ручной список чувствительностью

**Механизм.** У нас `Q6_ROUTED_DOWN_LAYERS = (34, 38, 39)` без записанного критерия
(`inventory.py:37`; `docs/maintainer/qwen3.6-35b-a3b-artifact.md:250`). NVIDIA формализовала
это как задачу распределения бюджета: чувствительность слоя
`S(Op_i, Q_f) ∝ Σ_k (g_{i,k})² (Y_{i,k} − Y^{Q_f}_{i,k})²` (диагональный Фишер вместо гессиана,
на **выходе** оператора), бюджет — «effective bits» (NVFP4 = 4.5, FP8 = 8, BF16 = 16), решение —
ILP (https://nvidia.github.io/Model-Optimizer/announcements/autoquantize.html). Ключевое
ограничение оттуда: **все sparse experts одного MoE-слоя — одно решение**, что ровно совпадает
с гранулярностью нашего инвентаря. Один backward-проход на батч, `O(N_layers × N_formats)`.

**Ожидаемый эффект.** Скорости не меняет; перераспределяет тот же бюджет байт. Ориентир Intel
AutoScheme (https://github.com/intel/auto-round/blob/main/docs/auto_scheme_acc.md): при среднем
5 бит наивные эвристики «хвостовые слои в 8 бит» 0.6671 и «головные слои в 8 бит» 0.6657 против
поиска **0.6857** (Llama-3.1-8B-Instruct); на 3 битах разрыв 0.3198 против 0.6148.
То есть ручной список — **измеримо хуже поиска**, и наши три слоя стоит перепроверить.

**Чем меряем.** Переназначить Q5/Q6 по чувствительности при том же суммарном размере артефакта
(±0.5 %) и сравнить PPL с текущим списком. Если PPL не сдвинулась — список (34, 38, 39)
подтверждён, и это тоже результат.

**Риски и что ломается.** Требует offline-прогона с градиентами (PyTorch, вне рантайма) —
инфраструктуры под это в `tools/` нет. Оценка чувствительности шумная: NVIDIA прямо пишет, что
кривая «не монотонна» из-за шума оценки.

**Оценка объёма.** Новый offline-скрипт (~400 строк) вне `src/`; правка одной строки инвентаря.

## Опровергнуто / не переносится

- **«Решает наличие квантования активаций» — подтверждено, но формулировку надо уточнить.**
  Для compressed-tensors различитель — `input_quant is None` (`compressed_tensors.py:736`,
  `compressed_tensors_moe.py:152`). Для ModelOpt — **литеральная строка** `"W4A16_NVFP4"` против
  `"NVFP4"` в `quantized_layers` (`modelopt.py:2393-2396`), а не отсутствие поля. Для MoE есть
  третий, независимый путь: cutlass-эксперты отвергают конфиг при `activation_key=None`
  (`cutlass_moe.py:724`), и перебор доходит до Marlin. Три разных механизма, один результат.
- **«nvidia/… — без квантования активаций».** Карточка NVIDIA для этой же модели гласит
  «Only the weights **and activations** of the linear operators within transformer blocks in MoE
  are quantized» (https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4). Это противоречит
  наблюдённому Marlin. Проверяемое предсказание: в `hf_quant_config.json` этой сборки у
  MoE-слоёв стоит `quant_algo: "W4A16_NVFP4"` (тогда карточка неточна), либо она W4A4, а Marlin
  выбран по другой причине. **Один `jq` по чекпоинту закрывает вопрос** — из клона это не видно.
- **Колонка «размер» в отчёте не может быть VRAM.** unsloth 26 ГБ даёт KV 333k, а RedHatAI
  24 ГБ — только 286k. Плюс Marlin-путь **увеличивает** резидентные веса на 11.1 % против
  дискового NVFP4 (скейл fp8→param_dtype, `marlin_utils_fp4.py:271`), так что «Marlin = самый
  маленький» подозрительно. Выводить из этой таблицы что-либо о занятости памяти нельзя.
- **Группа 16 для наших int-форматов — бессмысленна.** Группа 16 существует потому, что
  `mma.kind::mxf4nvf4 … scale_vec::4X.m16n8k64` берёт ровно 4 скейла на k=64
  (`ninfer/src/ops/common/mma.cuh:89`). Для int4-кодов тензорного ядра на sm_120 нет вовсе, коды
  разворачиваются в bf16 (`sparse_moe_prefill_kernels.cu:355`, `q4_rowsplit_storage.cuh:48-60`).
  Группа 16 для Q4 стоила бы
  16/16 = 1 бит скейла на вес (4 → 5 бит/вес, **+25 % байт**) и не дала бы ни одного FLOP.
- **Импорт GPTQ/AWQ-чекпоинтов «как есть» невозможен.** Оба формата асимметричные, с явными
  нулями (`auto_gptq.py:442-445` — `qzeros`; `auto_awq.py:492-494` — `qzeros`), наши
  `QuantFormat` симметричные с фиксированным `qmin/qmax` (`numeric.py:53-56`). Переносится
  алгоритм (GPTQ штатно работает с `sym=True`, `marlin/gptq/quant.py:38-47`), не байты.
- **«Не квантуйте голову» — к нам не применимо в исходном виде.** Совет единодушный
  (llm-compressor issue #1820: «we strongly recommend leaving `lm_head` unquantized»; RedHatAI
  держит `lm_head` в `ignore`; NVIDIA в arXiv:2509.25149 оставляет output head в исходной
  точности), но он про **4-битный int без выделенного формата**. Мы держим голову в Q6G64 /
  W8G32 / FP8-row (`inventory.py:98`, `inventory_nvfp4.py:147`) и GPQA не потеряли. Численной
  опубликованной дельты PPL от 4-битной головы не существует — веб-поиск её не нашёл.
- **AutoRound целиком не переносится.** Он обучает сдвиг округления `V` и границы клипа `α,β`
  через SignSGD, 200-1000 шагов на блок с backward
  (https://arxiv.org/abs/2309.05516), и пишет в чужую упаковку
  (`sglang/.../auto_round.py:50-52`). Стоимость на 35B-A3B с 256 экспертами × 40 слоёв высока,
  а выигрыш над «light»-режимом на 4 битах по их же таблице отрицательный (Qwen2.5-7B-Instruct
  W4G128: best 0.6426 против light 0.6453). **Брать надо GPTQ, не AutoRound.**
- **Онлайн-повороты (INPUT/OUTPUT/K_CACHE/Q_ATTN) — не наш случай.** Они требуют кернела на
  каждом форварде (`transform/factory/base.py:120-125, 158-165`). Адамар на K/Q у нас уже есть
  (`src/ops/kv_cache/hadamard_d256.cuh`). Новое здесь — **только offline-вариант** (кандидат 2).
- **«W4A4 быстрее» — не для декода.** На 5090 fp4-ярус 2021.8 против bf16 255.3, но это про
  префилл. Наш собственный диспатч уже уводит NVFP4 на A16 ниже T = 4/5/8
  (`nvfp4_dispatch.cpp:30-40`), потому что при малом T квантование активаций не окупается.
  Внешнее подтверждение того же (https://blog.nota.ai/insights/nvidia-blackwell-nvfp4).
- **W4A4 против W4A16 по точности — данных нет.** Единственное количественное утверждение —
  форумный пост «KLD scores show W4A4 is 2-4× worse than W4A16, especially past ~10K context»
  (https://forums.developer.nvidia.com/t/.../370403), без сырых чисел. Таблица unsloth
  (MMLU-Pro / GPQA / AIME25) разницы **не показывает вообще**. Косвенно за деградацию говорит
  паттерн RedHatAI: знаниевые бенчи recovery ≈ 100 %, а SWEBench Verified **91.61 %**,
  BFCLv4 Multi-Turn **93.38 %**, LCB Codegen 96.55 %
  (https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4). **Наша батарея (AIME25 /
  GPQA-diamond / LongBench) по этому паттерну увидит только LongBench.**
- **«AWQ»-сборки не обязательно AWQ.** Эталонный скрипт NVIDIA помечает веса
  `W4A8_AWQ`, а получает их обычным RTN `abs().max()/7`
  (`trtllm/examples/quantization/quantize_mixed_precision_moe.py:129-148`). Название алгоритма в
  карточке — не свидетельство.
- **Правило смешанной точности у всех разное и почти всегда ручное.** NVIDIA/DeepSeek: attention
  + shared experts + MLP **первых трёх** слоёв в FP8, routed experts 3..60 в 4 бита
  (`quantize_mixed_precision_moe.py:281-305`). unsloth/наш 27B: MLP 0-55 в NVFP4, MLP
  **последних восьми** слоёв в FP8 (`inventory_nvfp4.py:46-47`). Intel: 4 бита **только**
  routed experts, всё прочее 16 бит. RedHatAI: `ignore` = `lm_head`, `visual.*`, `mlp.gate`,
  `embed_tokens`, `shared_expert_gate`, `linear_attn.*`. Единственная общая инвариантa —
  **routed experts терпят 4 бита, attention и маршрутизаторы — нет**; она у нас уже соблюдена
  (attention/GDN в W8/FP8, `inventory.py:60-76`). Направление «скопировать чужое правило»
  закрыто: копировать нечего, правила противоречат друг другу.

## Открытые вопросы

1. Чем на самом деле выбраны слои 34, 38, 39 под Q6? В дереве критерия нет. Если это результат
   измерения — он не записан; если эвристика — по данным AutoScheme она, скорее всего, хуже поиска.
2. Какой энкодер использовал unsloth для `Qwen3.8-27B-NVFP4` — чистый RTN
   (`QuantizationModifier`) или GPTQ/AWQ поверх? Это определяет, на сколько именно наш RTN
   отстаёт. compressed-tensors содержит только формат; алгоритм живёт в llm-compressor, которого
   в корпусе нет. Ответ читается из `recipe.yaml` в самом репозитории unsloth.
3. Каков `quant_algo` в `hf_quant_config.json` у `nvidia/…-NVFP4`? От этого зависит, подтверждает
   отчёт правило диспатча или обнаруживает второй, ещё не найденный путь к Marlin.
4. Сколько применимых пар для offline-поворота реально есть в графе 35B-A3B (с учётом RMSNorm
   между блоками и маршрутизации MoE)? Без инвентаризации кандидат 2 не оценить.
5. Насколько плохо обусловлен гессиан для редких экспертов при разумном калибровочном бюджете?
   У palmfuture 2.58 % модулей ушли в RTN-фолбэк, но у них другая разбивка слоёв.
6. Публичных аблаций group_size 16/32/64/128 на 4 битах не существует. Ближайшее —
   NVFP4 против MXFP4 при обучении (относительная ошибка лосса 1.5 % против 2.5 %, MXFP4
   догоняет только при +36 % токенов, arXiv:2509.25149 §5) и GPTQ Table 7 на **2** битах
   (OPT-175B Wiki2: g128 9.58 → g64 9.18 → g32 8.94 при FP16 8.34). **Замер g32/g64/g128 на
   нашем формате был бы новым знанием, а не воспроизведением.**
