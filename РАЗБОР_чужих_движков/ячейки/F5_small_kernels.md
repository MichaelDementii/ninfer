# F5 — мелкие ядра: нормы, RoPE, активации, элементвайз, конверсии

## Что это за механизм

Все четыре чужих движка сходятся на одном принципе: **мелкое ядро не имеет права
материализовать промежуточный тензор в bf16, если следующий потребитель всё равно его
переформатирует**. Из этого у них вырастают три семейства слияний: (а) норма/активация/выход
внимания сразу выдают квантованный код нужного следующему GEMM формата; (б) residual-сложение
живёт внутри нормы; (в) пролог внимания (split QKV → нормы → RoPE → запись в страничный кеш с
конверсией) — одно ядро. vLLM довёл это до компиляторного каталога паттернов, TRT-LLM и
FlashInfer — до руками написанных мега-ядер.

У NInfer семейство (б) устроено **лучше**, чем у них (residual сидит в эпилоге GEMM, а не в
норме), семейство (в) экономически не окупается на нашей геометрии (2 KV-головы), а семейство
(а) — **единственная реальная дыра**: NInfer гоняет отдельное ядро квантования активаций
(`fp8_a8_quantize_kernel`, `nvfp4_w4a4_quantize_kernel`) сразу после нормы, которая только что
держала ту же строку в регистрах.

Второй результат этой ячейки — **количественный**: весь пул мелких ядер в префилле 35B-A3B
это ≈2–3% времени, и это подтверждено сходимостью трассы и арифметики (см. ниже). Почти всё,
что здесь можно слить, экономит меньше 0.5%. Кандидатов выше порога ровно два.

### Опорные числа, от которых считаю

| величина | значение | источник |
|---|---|---|
| пик HBM | 1792 ГБ/с | BRIEF |
| префилл 35B-A3B, чанк 8192, 32K | 20 174 ток/с → **49.6 мкс/токен** | `ninfer-prefill-mma-tier` |
| префилл 27B NVFP4, чанк 4096, 32K | 7 691 ток/с → **130.0 мкс/токен** | `ninfer-27b-nvfp4-transfer` |
| декод 35B mtp0 | 336.5 ток/с, **448 ядер/такт**, пузыри 13.6% → ≈0.9 мкс разрыва на ядро | `ninfer-experiment-log` |
| доля «норм» в префилле по трассе | **1.5%** | `ninfer-prefill-mma-tier` |

**Проверка модели.** Сложил трафик всех норм 35B-A3B руками при T-независимой геометрии
(hidden 2048, q_size 4096, kv_size 512, value_dim 4096, 10 GQA + 30 GDN):
post_attn_norm 40×8192 Б + input_norm 10×8192 + GDN input_norm 30×8192 (внутри
`gdn_norm_gating_proj`) + q_norm 10×16384 + k_norm 10×2048 + `gated_rmsnorm` 30×24576 +
final_norm 8192 = **1.585 МБ/токен** → 0.884 мкс → **1.78% префилла**. Трасса даёт 1.5%.
Расхождение объясняется тем, что часть норм слита в GEMM-ядра и не попадает в категорию
«нормы» трассы. Модель годится, дальше считаю ей.

---

## Доказательства из кода

### Таблица слияний: что чужие делают, а мы нет

Формат: «что слито | у кого | у нас | что экономит».

**1. RMSNorm → квантование активаций (fp8 / int8 / nvfp4 / блочное).**
- vLLM: `vllm/csrc/libtorch_stable/layernorm_quant_kernels.cu:22` `rms_norm_static_fp8_quant_kernel`;
  `:87` `fused_add_rms_norm_static_fp8_quant_kernel`;
  `vllm/csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu:49`
  (динамический per-token) и `:94` `rms_norm_per_block_quant_kernel` (группы 128/64);
  каталог паттернов `vllm/vllm/compilation/passes/fusion/rms_quant_fusion.py:118-138`.
- FlashInfer: `flashinfer/include/flashinfer/norm.cuh:175` `RMSNormQuantKernel`; `:544`
  `FusedAddRMSNormQuantKernel`; `:690` `FusedAddRMSNormFP8BlockQuantKernel` — **выдаёт четыре
  тензора за один пуск** (fp8-коды, fp32 блок-масштабы, bf16 residual, bf16 normed_out),
  строка живёт в регистрах (`float x[VEC_SIZE]` на `:712`), без shared memory.
- SGLang: `sglang/python/sglang/srt/layers/layernorm.py:904` и `:912` — зовёт именно эти
  flashinfer-ядра; комментарий на `:903` описывает механику «residual += x, затем
  out = quant(rmsnorm(residual) * w)».
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/rmsnormKernels.cu:136-141` (per-tensor) и `:153-176`
  (динамический per-token с `blockAllReduceMax`); `trtllm/cpp/tensorrt_llm/kernels/rmsNormFp4QuantKernels.cu:172`
  `rmsNormFp4QuantKernel` — residual+bias+норма+NVFP4+блок-масштабы, вход в регистрах.
- **У нас: отдельным ядром.** `ninfer/src/ops/linear/fp8/fp8_a8.cu:20` `fp8_a8_quantize_kernel`
  (один CTA на токен, читает всю bf16-строку, absmax по строке, пишет коды + масштаб),
  запускается на `:96-101`, вызывается из `launch_fp8_a8` на `:138` **перед** GEMM.
  То же для NVFP4: `ninfer/src/ops/linear/nvfp4/nvfp4_w4a4.cu:38-49` `launch_quantize_exact`.
  Производитель активации — норма: `ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h:825`
  (`input_norm`) и `:991` (`post_attn_norm`).
- **Экономит: 4 Б/элемент** (запись bf16 + чтение bf16). Пара «норма+квант» стоит сегодня
  6.56 Б/эл (nvfp4), слитая — 2.56 Б/эл, то есть **−61% трафика пары**.

**2. Активация (SwiGLU / act_and_mul) → квантование.**
- vLLM: `vllm/csrc/libtorch_stable/quantization/activation_kernels.cu:43` `act_and_mul_quant_kernel`;
  `vllm/csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu:39`
  `silu_mul_cvt_fp16_to_fp4`; `vllm/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu:15`;
  таблица `vllm/vllm/compilation/passes/fusion/act_quant_fusion.py:33-44`.
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:2300`
  `doActivationKernel` — act+mul+bias и сразу NVFP4/MXFP8/fp8-blockscale в эпилоге (`:2501-2537`);
  `trtllm/cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/fusedGatedRMSNormQuant.cu:144` —
  gated-SiLU × mul + групповая RMSNorm + NVFP4, документированный **один проход по HBM**
  (`:138-142`), конверсия f32→fp4 напрямую минуя bf16 (`cvt_float_to_fp4_inline`, `:75`).
- SGLang: `sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py:372`
  `silu_and_mul_masked_post_quant_fwd`.
- **У нас: отдельным ядром.** `ops::linear_swiglu` пишет bf16 —
  `ninfer/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu:62`
  (`out[gate_row] = __float2bfloat16_rn(silu(gate) * up)`), после чего down-проекция гоняет
  `nvfp4_w4a4_quantize_kernel` по этому же тензору.
- **Экономит 4 Б/эл** на ширине intermediate = 17408 (27B).

**3. Выход внимания → квантование.**
- vLLM: `vllm/vllm/compilation/passes/fusion/attn_quant_fusion.py:38` `AttnFp8StaticQuantPattern`
  и `:172` `AttnNvfp4QuantPattern` — масштаб уезжает аргументом `output_scale` в само ядро
  внимания, узел квантования удаляется из графа.
- **У нас: отдельным ядром** (тот же `fp8_a8_quantize_kernel` перед out-проекцией,
  `ninfer/src/targets/qwen3_6_27b/impl/variant.cpp:175` `ops::linear_add(..., text_policy(weight), ...)`).

**4. Гейтованная норма (SiLU-гейт × RMSNorm) в один регистровый проход.**
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/fusedGatedRMSNormQuant.cu:144`
  `fusedGatedRMSNormQuantKernelOptimized`, фаза 1 (`:179-212`) держит гейтованные значения
  в регистрах, фаза 2 (`:245-281`) нормирует их там же.
- **У нас: есть, но отдельным ядром и с двумя чтениями.**
  `ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h:985` `ops::gated_rmsnorm(o, w, z, eps, on)`;
  ядро `ninfer/src/ops/kernel/rmsnorm.cuh:24` (`RmsEpilogue::Gated`) читает `o`, читает `z`,
  пишет `on`. Производитель `o` — `ninfer/src/ops/linear_attention/gated_delta_net/chunked/output.cuh:330-334`.
- **Экономит 4 Б/эл** (запись `o` + чтение `o`), если норму убрать в эпилог GDN.

**5. Две нормы разных тензоров + конкатенация в один пуск.**
- SGLang: `sglang/python/sglang/kernels/jit/csrc/elementwise/fused_eh_norm.cuh:28`
  `fused_eh_norm_kernel` — нормирует `embeds` весом `enorm_weight` (`:47-51`) и
  `previous_hidden` весом `hnorm_weight` (`:53-56`) в одном блоке, переиспользуя тот же
  `smem` для обеих редукций, и пишет их **рядом**: `output` и `output + kHidden`. Это
  буквально вход EAGLE/MTP.
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/groupRmsNormKernels/groupRmsNormKernels.cu:91`
  `GroupRMSNormBaseKernel` — n норм **разной ширины** в одном пуске; ширины в
  `input_last_dims[n]` (`groupRmsNormKernels.h:32-47`), варпы распределяются между тензорами
  пропорционально ширине на хосте (`:583-618`), знаменатель считается отдельно для каждого
  тензора по его собственной ширине (`:205`).
- **У нас: три пуска.** `ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h:347`
  (`rmsnorm(emb) → e`), `:348` (`rmsnorm(hidden) → h`), `:351` (`ops::mtp_pack_fc_input(e, h, fc_in)`).
  Ядро упаковки — **чистая копия**: `ninfer/src/ops/kernel/mtp_pack.cuh:13-24`.
- **Экономит 8·H·T байт** (половина трафика стема) и 2 пуска.

**6. q-норма и k-норма с разным числом голов в одном пуске.**
- SGLang: `sglang/python/sglang/kernels/jit/csrc/elementwise/qknorm.cuh:38` (`fused_qknorm_warp`)
  и `:72` (CTA-вариант) — работа развёрнута по `num_qo_heads + num_kv_heads` (`:47`), тензор и
  вес выбираются по `head_id < num_qo_heads` (`:57-60`); указатель `k` смещён на хосте на
  `-2*num_qo_heads*kHeadDim` (`:153`), чтобы убрать вычитание из внутреннего цикла.
- vLLM: `vllm/csrc/libtorch_stable/layernorm_kernels.cu:36-42` (`NUM_DIMS==3`) — одно ядро
  нормирует каждую строку `(batch, head)`; плюс двумерная форма веса `[num_groups, hidden]`
  со `weight_stride` (`:23-26`), то есть **разные веса разным строкам в одном пуске**.
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu:97` — варп на голову,
  сегмент выбирается по `isQ`/`isV` (`:144-162`).
- **У нас: два пуска.** `ninfer/.../text_context_impl.h:841` и `:842`.
  *Пересекается с уже поставленной в очередь ячейкой* — не заявляю отдельно.

**7. split QKV + bias + (QK-норма) + RoPE + запись в KV-кеш с конверсией/квантованием.**
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h:339`
  (`applyBiasRopeUpdateKVCache`) — split (`:405-411`), bias (`:488-508`), RoPE (`:520-558`),
  запись в кеш int8/fp8 (`:637-643`) или NVFP4 (`:644-657`), маскирование паддинга (`:429-434`),
  всё в регистрах, единственный `__syncthreads()` на `:574` — только ради in-place перезаписи.
  **QK-нормы там нет** — она отдельным ядром `fusedQKNormRopeKernel.cu:97`.
- SGLang: `sglang/python/sglang/kernels/jit/csrc/inkling/inkling_attn_prologue_fused.cuh:144` —
  split q/k/v + пер-головная RMSNorm + conv1d + SiLU + residual + MXFP8 + scatter в страничный
  кеш; `static_assert(!USE_MXFP8 || DO_STORE, "MXFP8 prologue quantization owns the KV store")`
  на `:146`.
- vLLM: `vllm/csrc/libtorch_stable/cache_kernels_fused.cu:24`
  `concat_and_cache_mla_rope_fused_kernel` — поворот и запись в страницу в одной итерации цикла
  (`:110-143`); пассы `rope_kvcache_fusion.py:279` и `qk_norm_rope_kvcache_fusion.py:108`.
- FlashInfer: `flashinfer/include/flashinfer/pos_enc.cuh:808`
  `RopeQuantizeAppendPagedKVCacheKernel`.
- **У нас: раздельно.** `rmsnorm(k)` → `rope` → внутри внимания
  `ninfer/src/ops/softmax_attention/dense/causal_cache/prompt.cu:91`
  `kv_cache_append_batch_launch` (и `prompt_fp8.cu:69`) — это **отдельный пуск перед ядром
  внимания**, не эпилог. В MTP-префилле цепочка видна явно:
  `text_context_impl.h:494`, `:495`, `:496`.
- **Экономит 2560 Б/токен/слой → 0.029% префилла 35B.** Ниже порога, разбор ниже.

**8. Квантование + scatter в страничный кеш в одном ядре.**
- FlashInfer: `flashinfer/include/flashinfer/page.cuh:501` `NVFP4QuantizeAppendPagedKVCacheKernel`
  (обычный `AppendPagedKVCacheKernel` на `:395` конверсии **не делает** — чистый `memcpy`).
- vLLM: `vllm/csrc/libtorch_stable/cache_kernels.cu:264` `reshape_and_cache_flash_kernel`,
  `:445` `concat_and_cache_ds_mla_kernel` (динамический масштаб плитки считается прямо в ядре
  и кладётся внутрь записи кеша, `:509-528`), `:548` `indexer_k_quant_and_cache_kernel`.
- SGLang: `sglang/python/sglang/kernels/jit/csrc/attention/fused_fp8_qkv_kv_cache.cuh:59`.
- **У нас: УЖЕ ЕСТЬ и полнее.** `ninfer/src/ops/kv_cache/append/kernel.cuh:203`
  `kv_cache_append_full_i8_kernel` — Хадамар (`:232` `normalized_hadamard_d256_inplace`) +
  групповое int8-квантование (`:235-260`) + запись в страницу + масштабы, всё за один пуск;
  fp8-варианты на `:142` и `:170`, страничный i8 на `:267`.
- **Ничего не переносится.**

**9. causal conv1d + SiLU + residual add.**
- SGLang: `sglang/python/sglang/kernels/jit/csrc/inkling/causal_conv1d.cuh:55` — `USE_SILU` и
  `USE_RESIDUAL` шаблонные, SiLU на `:142-143` (`__fdividef` + `__expf`), residual на `:145-148`.
- TRT-LLM: `trtllm/cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu:60`, `:246`.
- **У нас: conv1d + SiLU + split q/k/v уже слиты** (`ninfer/src/ops/kernel/causal_conv1d.cuh:3`,
  наш PR #99). **Residual в GDN на этом месте нет вообще** — в Qwen3.6 после свёртки идёт
  рекуррентность, а не сложение. Не применимо.

**10. bias в эпилог GEMM.**
- Все четыре движка. У нас `ops::linear` без bias-эпилога, в vision-башне 7 отдельных пусков
  `add_bias` на блок: `ninfer/src/targets/qwen3_6/impl/runtime/vision_context_impl.h:301`, `:328`,
  `:353`, `:367`, `:370`, `:384`, `:387`. Плюс 2 `layer_norm`, 1–2 `gelu`, 2 `residual_add`.
  Vision-only.

**11. Отдельные ядра-касты.**
- FlashInfer: 4 периферийных (`flashinfer/csrc/trtllm_fused_moe_kernel_launcher.cu:843`,
  `flashinfer/csrc/fp4_kv_dequantization.cu:53`, `:131`, и дескейл в MoE); всё горячее свёрнуто
  в `cast_load`/`cast_store` (`flashinfer/include/flashinfer/vec_dtypes.cuh:752-773`), которые
  компилируются в ноль при совпадении типов.
- vLLM: **ровно один**, `convert_fp8_kernel` (`vllm/csrc/libtorch_stable/cache_kernels.cu:954`),
  и он помечен «Only for testing» на `:973`.
- TRT-LLM: 12 отдельных (`trtllm/cpp/tensorrt_llm/common/memoryUtils.cu:235`, `:510`, `:544`,
  `:620`; `kernels/quantization.cuh:33`, `:188`; `common/cudaFp8Utils.cu:57`, `:105`, `:246`,
  `:528`; `kernels/quantization.cu:438`, `:452`).
- **У нас:** `ninfer/src/ops/kernel/cast.cuh` — три ядра fp32→bf16, и `grep ops::cast` по
  `src/targets` не находит **ни одного вызова**. Мёртвый код, не расход.

### Сравнение реализаций самих ядер (не границ)

| признак | NInfer | vLLM | SGLang | FlashInfer | TRT-LLM |
|---|---|---|---|---|---|
| RMSNorm: строка после редукции | **в регистрах** (`rmsnorm.cuh:46`, `:138`); повторное чтение из HBM только в fallback `:232-253` | в регистрах (`_f16Vec`, `type_convert.cuh:104-195`) | в регистрах (`fused_add_rmsnorm.cuh:69`) | **перечитывает из HBM** в `RMSNormKernel` (`norm.cuh:86` → `:122`); в смем в `FusedAddRMSNorm` (`:456`→`:490`); в регистрах только в fp8-block-варианте (`:712`) | смем при `USE_SHMEM`, иначе **перечитывает и пересчитывает** (`rmsnormKernels.cu:161-165`) |
| редукция | fp32, warp-shuffle + `block_reduce_sum` | fp32 `cub::BlockReduce<float,1024>` | fp32, `cooperative_groups::reduce` | fp32, ручной shfl-бабочка, без cub | fp32 `blockReduceSumV2` |
| вектор | bf16x2 (4 Б) в warp/CTA-маршрутах; `uint4` только в PR-кандидате d5120 | `gcd(16/sizeof, hidden)` → 16 Б | `kMaxVecBytes` 16/32 Б (32 Б на Blackwell) | `gcd(16/sizeof, d)` → 16 Б; 32 Б в fp8-block | **фиксированные 2 элемента (4 Б)** (`rmsnormKernels.cu:240`) |
| хвост | маскированные ветки `if (pair < pairs)` | `gcd` гарантирует делимость | **отказ**: `RuntimeCheck(hidden % vec == 0)` (`fused_add_rmsnorm.cuh:174`) | маскированный векторный раунд (`norm.cuh:85`) | скаляр при нечётном hidden |
| ветка для узких строк | **есть**: `rmsnorm_warp_bf16x2_kernel` для D∈{64..256}, `rmsnorm_d128` (`launcher/rmsnorm.cu:23`, `:34`) | нет отдельной, только `max_block_size` 1024/256 | warp при `kDim ≤ 256` (`impl/norm.cuh:39`) | `QKRMSNormKernel` варп-на-строку (`norm.cuh:291`) | нет (CTA на строку всегда) |
| экспонента | **точный `expf`** везде (`ninfer/src/ops/common/math.cuh:13`, `:15`, `:17`), политика задекларирована в `silu_and_mul.cuh:5-6` | `expf` в обычном пути, `__fdividef` в deep-gemm (`quantization/activation_kernels.cu:111`) | расщеплено по происхождению: `expf` в mamba-ядрах (`jit/csrc/mamba/causal_conv1d.cuh:282`), `__expf`+`__fdividef` в новых inkling (`jit/csrc/inkling/causal_conv1d.cuh:142`) | **`__expf`** для silu (`flashinfer/jit/activation.py:88`), точный `::erf` для gelu | **`__expf`** + `reciprocal_approximate_ftz` (`fusedGatedRMSNormQuant.cu:58`, `:64`) |
| LayerNorm: дисперсия | **Уэлфорд с делением на элемент** (`ninfer/src/ops/kernel/layer_norm.cuh:45-50`) | сумма+сумма квадратов | — | `float2`-редукция `E[x²]−E[x]²` с клампом (`fused_dit_layernorm.cuh:693-702`) | `USE_DIFF_OF_SQUARES` переключатель (`layernormKernels.cu:47-55`) |

### RoPE на части измерений (64 из 256)

- **NInfer уже на полу и лучше FlashInfer.** `ninfer/src/ops/kernel/rope.cuh:106`:
  `if (lane >= kHalfPair) { return; }` — неповорачиваемые 192 из 256 измерений **не читаются
  и не пишутся вообще**, операция in-place. `kHalf = 32` пар = 64 измерения (`rope.cuh:124-126`).
- **FlashInfer грузит и пишет всю голову**: `flashinfer/include/flashinfer/pos_enc.cuh:133`
  — `vec.cast_load(...)` безусловный, поворот внутри `if (threadIdx.x*vec_size < rotary_dim)`
  (`:135`), а запись на `:341`/`:407` тоже безусловная. То есть на нашей геометрии
  FlashInfer двигал бы **в 4 раза больше байт**, чтобы повернуть те же 64 измерения.
- **TRT-LLM подставляет единичный поворот** (`cos=1, sin=0`) для хвостовых измерений и всё
  равно их пишет: `trtllm/.../unfusedAttentionKernels_2_template.h:266`, `:821`. Это осмысленно
  только потому, что у них RoPE слит с записью в KV-кеш и хвост писать надо в любом случае.
  GPT-J-ветка у них хвост не пишет (`:298-302` — запись под предикатом).
- **vLLM ровно как мы**: чистый in-place, граница цикла `nq = num_heads * embed_dim` при
  `embed_dim = rot_dim/2` (`vllm/csrc/libtorch_stable/pos_encoding_kernels.cu:49`, `:53`);
  дополнительно есть `rope_dim_offset` (`:47`, `:57`), позволяющий крутить не префикс, а
  срез в середине головы.
- **SGLang срезает на Python-уровне**: `sglang/python/sglang/srt/layers/rotary_embedding/base.py:375-377`
  — `q_rope = q_rope[..., :rotary_dim]`, хвост не трогается; материализующий `torch.cat`
  остался только в native-fallback (`:261-273`).
- Синус/косинус: у нас `sincosf` считается на лету и кешируется в shared на CTA-токен
  (`rope.cuh:130-136`) — 32 значения на токен, делятся между 18 головами. FlashInfer держит и
  таблицу, и вычисление на лету (`pos_enc.cuh:119` vs `:325-326`); TRT-LLM V2 перешёл на
  таблицу из памяти именно потому, что ядро bandwidth-bound (`unfusedAttentionKernels_2_template.h:710`).
  У нас ядро тоже bandwidth-bound, но таблица не окупится: 32 float на токен против 4608 Б
  трафика — доля трансцендентных операций уже ничтожна.

---

## Что у нас сегодня

Порядок мелких ядер на слой (35B-A3B, `ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h`):

**GQA-слой (10 из 40):** `rmsnorm(input_norm)` `:825` → `attn_input_proj` → `rmsnorm(q_norm)` `:841`
→ `rmsnorm(k_norm)` `:842` → `rope` `:848` → `kv_cache_append` (внутри внимания,
`prompt.cu:91`) → внимание → `sigmoid_mul` `:874` → `linear_add` (residual в эпилоге) →
`rmsnorm(post_attn_norm)` `:991` → MoE с `AddResidual`.

**GDN-слой (30 из 40):** `gdn_norm_gating_proj` (норма+проекция гейта слиты) `:888` →
`gdn_input_proj` → `causal_conv1d_silu_split` `:936` → `gated_delta_net` → `gated_rmsnorm` `:985`
→ `linear_add` → `rmsnorm(post_attn_norm)` `:991` → MoE.

Уже слито у нас и не является находкой: residual в эпилог GEMM
(`ninfer/src/targets/qwen3_6_35b_a3b/impl/variant.cpp:136` `linear_add`, `:73`
`SparseMoeEpilogue::AddResidual`); норма+гейт-проекция GDN (`:211`); conv1d+SiLU+split;
квант+Хадамар+scatter в страничный кеш; гейтованная норма как эпилог RMSNorm
(`rmsnorm.cuh:24`); подъём чтения весов выше редукции (PR #150).

Разбивка трафика мелких ядер по модели (35B, доля от 49.6 мкс/токен):

| ядро | Б/токен | мкс/токен | % префилла |
|---|---:|---:|---:|
| `gated_rmsnorm` (30 GDN) | 737 280 | 0.411 | **0.83%** |
| `post_attn_norm` (40) | 327 680 | 0.183 | 0.37% |
| GDN `input_norm` (30, внутри слитого ядра) | 245 760 | 0.137 | 0.28% |
| `sigmoid_mul` (10) | 245 760 | 0.137 | 0.28% |
| `q_norm` (10) | 163 840 | 0.091 | 0.18% |
| GQA `input_norm` (10) | 81 920 | 0.046 | 0.09% |
| `rope` (10) | 46 080 | 0.026 | 0.05% |
| `k_norm` (10) | 20 480 | 0.011 | 0.02% |
| **весь пул** | **≈1.87 МБ** | **1.04** | **≈2.1%** |

Это и есть потолок всей ячейки на 35B при чанке 8192.

---

## Кандидаты для NInfer

### 1. Квантование активаций в эпилоге производителя (норма → коды)

**Механизм.** `ops::rmsnorm` с шаблонным эпилогом, который вместо bf16 пишет код нужного
следующему GEMM формата (nvfp4-группа 16 / fp8 per-token / int8 per-group-32) плюс масштаб.
Ядро нормы **уже** держит всю строку в регистрах (`ninfer/src/ops/kernel/rmsnorm.cuh:46`,
`:138`, `:191-195`) и **уже** делает полнострочную редукцию — добавление второй редукции
(absmax по строке для fp8, absmax по группе 16 для nvfp4) стоит одного дополнительного дерева
shuffle. Прототип формы: FlashInfer `norm.cuh:690` выдаёт четыре тензора за проход, включая и
bf16-копию, и fp8-коды — это ровно тот режим, который нужен на 35B, где одна и та же `h`
кормит и W8-эксперта, и Q4-маршрутизируемых.

**Ожидаемый эффект.**
- *27B NVFP4, префилл.* Две нормы на слой шириной 5120 × 64 слоя = 655 360 элементов/токен.
  Экономия 4 Б/эл = **2.62 МБ/токен** = 1.46 мкс при 1792 ГБ/с = **1.12% префилла**
  (130.0 мкс/токен). Плюс 128 пусков на чанк.
  Если довести принцип до всех четырёх точек квантования (норма ×2, SwiGLU→down при K=17408,
  выход внимания/GDN→out-proj при K=6144): 8.65 МБ/токен = 4.83 мкс = **3.7% префилла**.
- *35B-A3B.* Прямой выгоды по байтам нет — маршрут `A16Only`
  (`ninfer/src/targets/qwen3_6_35b_a3b/impl/variant.cpp:264`). Но это **условие достижимости
  измеренного +7.3%** на `w8_rowsplit` в int8 (`ninfer-prefill-mma-tier`, поправка 2026-08-26:
  «достижим только если вынести квантование `x` из ядра»). Отдельное вынесенное ядро стоит
  3 Б/эл (чтение 2 + запись 1) = 0.49 МБ/токен = **0.55% префилла**; в эпилоге нормы оно стоит
  **−1 Б/эл** (int8 вместо bf16 на записи), то есть выигрывает ещё 0.18%. Разница между
  «вынести отдельным ядром» и «вынести в норму» = **0.73% префилла**, и это до основного +7.3%.
- *Декод.* По байтам ничтожно; по пускам — 2 ядра на слой × 64 слоя = 128 из 448 ядер такта на
  27B; при ≈0.9 мкс разрыва на пуск это верхняя оценка ≈115 мкс из 5.56 мс такта.
  Оценку разрыва брать с осторожностью — она выведена из общего числа 13.6% на 35B/mtp0.

**Чем меряем.** Стенд: две чередующиеся ноги одного дерева, 27B NVFP4, чанк 4096, промпт 32K,
int8-KV (канон из `ninfer-27b-nvfp4-transfer`); контроль дрейфа машины ≤4% за кампанию.
A/B: (A) master; (B) `rmsnorm` с эпилогом NVFP4, `launch_quantize_exact` для этих двух точек
не вызывается. Приёмка численности: вывод должен быть **побитово равен** A, потому что коды
получаются из тех же fp32-значений тем же округлением — если разошёлся, значит порядок
редукции absmax отличается, и это надо чинить, а не принимать. Второй прибор — операторный бенч
на паре (норма, квант) с отчётом доли от 1792 ГБ/с, как требует `ninfer-roofline-requirement`.
Порог клоков помнить (`ninfer-operator-bench-clock-floor`).

**Риски и что ломается.**
- Регистровое давление: absmax по строке — это ещё один аккумулятор и ещё одно дерево shuffle.
  По переписи `cuobjdump -res-usage` из `ninfer-rmsnorm-part-split` гейтованный эпилог при
  Block=512 уже стоит перехода 3→2 блока на SM; новый эпилог надо мерить тем же прибором,
  иначе повторим ошибку широкого гейта.
- Двойной выход (bf16 + коды) нужен там, где норму читают два потребителя разных форматов
  (35B: W8-шара и Q4-маршрут). Это +1 Б/эл на запись — всё равно втрое дешевле отдельного
  прохода, но форма API усложняется.
- Побитовость: при fp8 per-token масштаб зависит от порядка вычисления absmax. Порядок в
  `fp8_a8_quantize_kernel:49-56` (warp_max → smem → warp_max) и в норме
  (`block_reduce_sum` по warp_sums) **разный**. Это надо согласовать явно, иначе вывод поедет.
- Форматы: nvfp4-группа 16 требует, чтобы 16 соседних элементов строки лежали у согласованного
  набора нитей. В `rmsnorm_d2048_bf16x2_kernel` нить `t` держит пары `t` и `t+512`, то есть
  группа 16 = 8 соседних нитей — редукция на четверть варпа, дёшево. Для `d5120` (27B) надо
  перепроверить раскладку.

**Оценка объёма.** 4–6 файлов: `src/ops/kernel/rmsnorm.cuh` (новый эпилог),
`src/ops/launcher/rmsnorm.cu` (маршрут), `src/ops/wrapper/rmsnorm.cpp` (сигнатура/валидация),
`include/ninfer/ops/rmsnorm.h`, плюс места вызова в
`src/targets/qwen3_6/impl/runtime/text_context_impl.h` и `qwen3_6_27b/impl/variant.cpp`.
Ядра квантования при этом **не удаляются** — они остаются для точек, где производитель не норма.

### 2. `gated_rmsnorm` в эпилог ядра вывода GDN

**Механизм.** Ядро `gated_delta_net` chunked-вывода пишет `o` в HBM
(`ninfer/src/ops/linear_attention/gated_delta_net/chunked/output.cuh:330-334`), после чего
`ops::gated_rmsnorm` (`text_context_impl.h:985`) читает `o` обратно, читает `z` и пишет `on`.
Норма идёт по строке ширины `gdn_v_dim = 128` — это **ровно одна голова**, то есть ровно то,
что ядро GDN и производит. Прототип: TRT-LLM `fusedGatedRMSNormQuant.cu:144` держит
гейтованные значения в регистрах и нормирует их там же, документируя один проход по HBM
(`:138-142`).

**Ожидаемый эффект.** Экономия 4 Б/эл (запись `o` + чтение `o`) на `value_dim` элементов:
- 35B: 4096 эл × 30 слоёв × 4 Б = **491 КБ/токен** = 0.274 мкс = **0.55% префилла**;
- 27B: 6144 эл × 48 слоёв × 4 Б = **1.18 МБ/токен** = 0.659 мкс = **0.51% префилла**.
Плюс 30 (48) пусков на токен в декоде. Это **самый крупный одиночный элемент пула норм**
(0.83% на 35B — почти половина всего пула).

**Чем меряем.** Операторный бенч по `gated_delta_net` + `gated_rmsnorm` как паре, с долей от
1792 ГБ/с; затем сквозной префилл 32K на 35B (чанк 8192) и на 27B (чанк 4096), две чередующиеся
ноги. Побитовость: порядок суммирования квадратов внутри головы изменится (сейчас
`rmsnorm_warp_bf16x2_kernel` при D=128 суммирует по парам лейна, в GDN-эпилоге фрагмент MMA
раскладывает 128 измерений иначе) — **вывод не будет побитовым**, значит защищать надо
префиллом, где спекуляции нет (тот же довод, что для части 2 из `ninfer-rmsnorm-part-split`).

**Риски и что ломается.** Главный: **это не бесплатно**. `D_PANEL = 16`, `N_D_PANELS = 8`
(`output.cuh:47-49`) — 128 измерений головы производятся восемью последовательными панелями,
каждая со своим `cp.async`-этапом V. Чтобы нормировать строку, надо либо удержать 8×4 = 32
лишних float на нить, либо застейджить BT×128 bf16 ≈ 16 КБ в shared. Ядро уже многоэтапное,
и бюджет 99 КиБ/CTA придётся пересчитать. Второй риск: `linear_add`/`gdn_output_projection`
читает `on`, а не `o` — путь замены надо провести и через verify/replay-формы
(`gdn_replay_record`), иначе разъедется откат состояния при отказе.

**Оценка объёма.** 3–4 файла в `src/ops/linear_attention/gated_delta_net/chunked/` плюс место
вызова. Работа существенно тяжелее кандидата 1 при вдвое меньшем эффекте — поэтому вторым.

### 3. Стем MTP: две нормы + упаковка в один пуск

**Механизм.** Точный аналог SGLang `fused_eh_norm.cuh:28`. Минимальная версия вообще не требует
нового ядра: дать `rmsnorm` **шаг выходной строки**, чтобы `:347` писала в `fc_in[0:H]`, а `:348`
— в `fc_in[H:2H]`; `ops::mtp_pack_fc_input` (`src/ops/kernel/mtp_pack.cuh:13-24`, чистая копия)
удаляется целиком. Полная версия — один пуск на 2T строк с выбором источника и веса по индексу
строки, как в TRT-LLM `groupRmsNormKernels.cu:119-133`.

**Ожидаемый эффект.** Трафик стема 16·H·T → 8·H·T байт, пусков 3 → 2 (минимальная версия) или
3 → 1 (полная). На 35B при H=2048, T=1: 16 КБ и 2 пуска за черновой шаг, 6 пусков за раунд
mtp3. **Меньше 0.1% декода** — заявляю честно как под-пороговый, ценность в удалении целого
ядра и в том, что это самый дешёвый способ проверить механизм «две нормы в один пуск» перед
тем, как применять его к q/k.

**Чем меряем.** Декод 35B mtp3, восемь промптов; но прибор грубее эффекта (σ ≈ 2.7 пункта
приёмки, `ninfer-rmsnorm-part-split`), поэтому защищать надо счётом ядер в трассе nsys
(448 → 446 на такт mtp0) и операторным бенчом стема, а не сквозной цифрой.

**Риски.** Побитовость сохраняется (порядок редукции внутри каждой нормы не меняется).
Выходной шаг ломает предположение о непрерывности в валидаторе обёртки — придётся ослабить
контракт `rmsnorm`, что автор может не принять; безопаснее оформить как отдельный op
`rmsnorm_pair_concat`.

**Оценка объёма.** 2–3 файла.

### 4. `sigmoid_mul` в эпилог ядра внимания — ниже порога, но пусков много

`ops::sigmoid_mul(gate, a)` (`text_context_impl.h:874`, `:408`, `:541`; ядро
`src/ops/kernel/sigmoid_gate_mul.cuh:36`) читает `gate`, читает `a`, пишет `a` — три прохода по
тензору `q_size × T`. Если ядро внимания применит гейт в своём эпилоге, останется одно чтение
`gate`. Экономия 4 Б/эл: 35B 4096 эл × 10 слоёв × 4 Б = 164 КБ/токен = **0.18% префилла**;
27B 6144 × 16 × 4 = 393 КБ/токен = **0.17%**. **Ниже 0.5% — так и записываю.** Единственный
неарифметический аргумент: 10 (16) пусков на токен в декоде. Прототип наш собственный
(`RmsEpilogue::Gated`), внешний — vLLM `attn_quant_fusion.py:38`, где в эпилог внимания уже
уводят масштаб квантования, то есть механизм «эпилог внимания принимает поэлементную работу»
у них принят.

### 5. Vision-башня — отдельный, изолированный пакет

7 пусков `add_bias` + 2 `layer_norm` + 2 `gelu` + 2 `residual_add` на блок
(`vision_context_impl.h:301-387`). Все четыре чужих движка кладут bias в эпилог GEMM.
Плюс наш `layer_norm` считает дисперсию по Уэлфорду с **делением на каждый элемент**
(`src/ops/kernel/layer_norm.cuh:45-50`), тогда как TRT-LLM (`layernormKernels.cu:47-55`) и
FlashInfer (`fused_dit_layernorm.cuh:693-702`) берут `E[x²]−E[x]²` одной `float2`-редукцией.
Эффекта на текстовых бенчмарках нет вовсе; выносить только если vision станет предметом замера.

---

## Опровергнуто / не переносится

**`fused_add_rms_norm` — у нас лучше, переносить нельзя.** Это главное слияние всех трёх
open-source движков (vLLM `layernorm_kernels.cu:106`, FlashInfer `norm.cuh:414`, SGLang
`fused_add_rmsnorm.cuh:57`), и оно **хуже нашей схемы**. У них: GEMM пишет дельту (2 Б/эл),
затем слитое ядро читает дельту (2) + читает residual (2) + пишет residual (2) + пишет
нормированное (2) = **10 Б/эл после GEMM**. У нас: эпилог GEMM читает residual (2) и пишет
residual (2), затем норма читает residual (2) и пишет нормированное (2) = **8 Б/эл**
(`ninfer/src/targets/qwen3_6_35b_a3b/impl/variant.cpp:136`, `:73`). Отдельный `residual_add`
у нас выжил ровно в двух местах: vision (`vision_context_impl.h:354`, `:371`) и MTP-MLP на 27B
(`qwen3_6_27b/impl/variant.cpp:316`).

**Пролог внимания «k-норма + RoPE + Хадамар + квант + scatter» — не окупается на нашей
геометрии.** Это самое амбициозное чужое ядро (TRT-LLM `unfusedAttentionKernels_2_template.h:339`,
SGLang `inkling_attn_prologue_fused.cuh:144`, vLLM-пасс `qk_norm_rope_kvcache_fusion.py:108`).
Считаю для 35B: `kv_size = 512` элементов на токен. Сегодня цепочка стоит
1024 (чт. k) + 1024 (зап. kn) + 256+256 (RoPE) + 1024 (чт. kn) + 1024 (чт. v) + 1024 (зап. кодов)
+ 32 (масштабы) = 5664 Б/токен/слой; полностью слитая — 3104 Б. Экономия 2560 Б × 10 слоёв =
25.6 КБ/токен = 14.3 нс = **0.029% префилла**. Причина расхождения с ними прямая: у Qwen3.6
всего **2 KV-головы** и головы делят кеш-строку 512 элементов, тогда как у моделей, под которые
это писалось, KV-тракт в разы шире (MLA — сотни измерений на токен). Сам механизм верен,
арифметика на нашей модели его убивает. **Не заявлять без новой геометрии.**

**RoPE на части измерений — брать нечего, мы впереди.** `rope.cuh:106` не трогает 192 из 256
измерений. FlashInfer на той же задаче двигал бы вчетверо больше байт (`pos_enc.cuh:133`, `:341`).
vLLM ровно как мы. TRT-LLM пишет хвост, но только потому, что RoPE у них слит с записью в кеш.

**Квант + scatter в страничный кеш — уже есть, и полнее.** `kv_cache_append_full_i8_kernel`
(`src/ops/kv_cache/append/kernel.cuh:203`) делает Хадамар + групповое int8 + страницы + масштабы
одним ядром. Обычный `AppendPagedKVCacheKernel` FlashInfer (`page.cuh:395`) конверсии не делает
вообще.

**`__expf` вместо `expf` — не окупается.** Все перечисленные ядра (`silu_and_mul`,
`sigmoid_gate_mul`, `gdn_gating`) memory-bound: `sigmoid_mul` при 4096 элементах на строку и
3 Б/эл трафика тратит на трансцендентную часть доли процента. Экономия инструкций там, где
ядро ждёт HBM, равна нулю, а политика точности задекларирована в самом коде
(`silu_and_mul.cuh:5-6`, `gdn_gating.cuh:4`) и в `docs/op-development.md §6`. Единственное
место с реальной плотностью трансцендентных — `gdn_gating_kernel`
(`src/ops/kernel/gdn_gating.cuh:24-26`: `softplus` = `log1pf(expf(x))`, плюс `expf(A_log[h])`
и `i % 48` **на каждый элемент**), но у SGLang там ровно та же защищённая формула
(`fused_gdn_gating.py:35-37`), и на 35B этот путь и так слит в `gdn_norm_gating_proj`
(`variant.cpp:211`), то есть отдельным ядром не идёт.

**Компиляторные пассы vLLM — вне области.** `vllm/vllm/compilation/passes/fusion/*` это
ценность Python-уровня (сопоставление подграфов FX). Переносится только каталог того,
**что именно стоит сливать** — он и использован выше как карта. Сам механизм в AOT-движке
без JIT не воспроизводим и по брифу вне области.

**Все multi-GPU-слияния отброшены без разбора**: `AllReduceFusionPass`
(`allreduce_rms_fusion.py:1074`), `SequenceParallelismPass`, `AsyncTPPass`, TRT-LLM
`allReduceFusionKernels.cu:237`, SGLang `tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group`
(`layernorm.py:310`). Заметно, что у всех троих **самое плотное слияние нормы живёт именно в
all-reduce-эпилоге** — то есть их лучший образец этого класса нам недоступен по построению.

**MTP-MLP на 27B идёт нефьюженным путём — но это дыра реестра, а не места вызова, и она
под-пороговая.** `qwen3_6_27b/impl/variant.cpp:305-316`: `linear` → `silu_mul` → `linear` →
`residual_add`, тогда как основной `post_mixer` на `:296-302` использует
`linear_swiglu` + `linear_add`. Проверил, почему: веса MTP-MLP — `W8G32_F16S` формы
{34816, 5120} и {5120, 17408} (`qwen3_6_27b/impl/load/bindings.cpp:470`, `:473`), а
`linear_swiglu` допускает W8 только при K=2048/N=12288
(`src/ops/wrapper/linear_swiglu.cpp:84`), а `w8_linear_add_admits` — только при
rows=2048 и k∈{4096,6144} (`src/ops/linear_add/w8/w8_linear_add_plan.cpp:193-195`).
То есть нужен новый W8-маршрут, а не переключение вызова. Экономия ≈156 КБ на столбец и
2 пуска на черновой шаг → **≈0.03% раунда mtp3**. Ниже порога, но стоит записать как долг:
это единственное место, где `ops::linear` вызван **без политики**, то есть MTP-MLP на 27B
идёт по `A16Only` и не пользуется даже маршрутом A8, хотя веса это допускают.

**`src/ops/kernel/cast.cuh` — мёртвый код.** Три ядра fp32→bf16, `ops::cast` не вызывается
ни разу в `src/targets`. У vLLM единственное отдельное ядро-каст помечено «Only for testing»
(`cache_kernels.cu:973`); у FlashInfer конверсии свёрнуты в `cast_load`/`cast_store`, которые
компилируются в ноль при совпадении типов (`vec_dtypes.cuh:752-773`). Ни расхода, ни находки.

---

## Открытые вопросы

1. **Реальная цена одного пуска в графе декода.** Число 0.9 мкс выведено делением 13.6% пузырей
   на 448 ядер (`ninfer-experiment-log`) и относится к 35B/mtp0. Для 27B (64 слоя) число ядер
   не замерено. Без него все «−N пусков» оценки — верхние границы, а не результаты.
2. **Согласование порядка absmax.** Даст ли эпилог нормы **побитово тот же** fp8-код, что
   `fp8_a8_quantize_kernel:49-56`? Порядок редукции разный. Пока не проверено, все заявки на
   побитовость по кандидату 1 условны.
3. **Раскладка nvfp4-групп в `rmsnorm` при d=5120.** Группа 16 требует согласованного набора
   нитей; для 2048 раскладка проверена по коду, для 5120 (кандидат «часть 2» из
   `ninfer-rmsnorm-part-split`, ядро с `uint4`) — нет.
4. **Бюджет shared/регистров ядра вывода GDN.** `output.cuh` уже многоэтапное с `cp.async`;
   помещаются ли туда 16 КБ стейджа или 32 лишних регистра на нить — по коду не следует,
   нужен `cuobjdump -res-usage`.
5. **Отдельный `kv_cache_append_batch_launch` перед вниманием.** Это пуск, а не эпилог
   (`prompt.cu:91`). Почему он не свёрнут в первый k-тайл ядра внимания — из кода не видно;
   возможно, из-за формы batch/valid_columns. Стоит ли — см. отрицательный результат выше
   (0.029%), но вопрос о причине остаётся.
6. **Двойной выход нормы (bf16 + коды) на 35B.** Нужен ли он вообще, зависит от того, пойдут
   ли Q4-маршрутизируемые эксперты на int8-активации вместе с W8, или останутся A16. Замер
   `ninfer-prefill-mma-tier` даёт потолок для обоих (+18.5% MoE, +7.3% W8), но не отвечает,
   можно ли им дать один и тот же формат активации.
