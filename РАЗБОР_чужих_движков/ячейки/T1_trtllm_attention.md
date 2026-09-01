# T1 — ядра внимания TensorRT-LLM: что там есть для sm_120 и для head_dim 256

## Что это за механизм

TensorRT-LLM держит **четыре независимых семейства** ядер внимания, и только два из них
вообще собираются под sm_120. Контекст (префилл) — генератор `fmha_v2` (кубины + один
рукописный TU); декод — `xqa` (`mha.cu`) и скалярный MMHA. Датацентровое семейство
`trtllmGenKernels/fmha` включается только для sm_100 (`isSM100Family()`), Hopper-ядра
(`warpspec/`, `flashMLA`) требуют `wgmma` и на нашей карте не существуют.

Для нас существенно ровно три вещи, и все три подтверждены кодом:
1. **head_dim 256 у них первоклассный случай**, а не край: под sm_120 генерируются
   bf16/fp16/e4m3 ядра на D=256, и есть отдельный рукописный `d256` мост.
2. **Схема при большой голове — не «раздвинуть smem», а разрезать саму контракцию по D**:
   Q/K живут в smem кусками по 64 элемента головы, V — кусками по 32 ключа × 256 (или
   32 × 64 в warpspec-варианте). Полная голова в smem не лежит никогда.
3. **Пониженная точность аккумулятора у них есть, отгружена и является умолчанием для
   fp16-моделей** — и BMM1, и BMM2. Защита от переполнения — не масштабирование, а
   (а) насыщающий клип на стороне P и (б) нормировка O на каждом шаге KV, а не в конце.

---

## Доказательства из кода

### 1. Инвентаризация семейств и что из них живёт на sm_120

| семейство | файл | арх | наш случай |
|---|---|---|---|
| контекстный FMHA (генерируемый) | `trtllm/cpp/kernels/fmha_v2/setup.py:6797-6816` | 70/75/80/86/89/90/100/**120** | **да** |
| контекстный FMHA, «tiled» (гранулярный) | `trtllm/cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_noloop_tiled.h:31` | sm_mma 80 ⇒ и 120 | **да, это дефолт при D≥256** |
| контекстный warp-spec Hopper | `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec/compute.h` | только sm_90 (`wgmma`) | нет |
| контекстный warp-spec **sm_120/121** | `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md:1-13` | **только sm_120/121** | да |
| TRT-LLM-Gen FMHA | `trtllm/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp:55` — `mUseTllmGen = isSM100Family()` | только sm_100/103 | **нет** |
| XQA (декод) | `trtllm/cpp/kernels/xqa/mha.cu:92` — `__CUDA_ARCH__ == 860 \|\| 890 \|\| 1200 \|\| 1210` | 86/89/**120/121** | да |
| XQA Hopper | `trtllm/cpp/kernels/xqa/mha_sm90.cu` (gmma) | sm_90 | нет |
| XQA MLA sm_120 | `trtllm/cpp/kernels/xqa/mla_sm120.cu:20` — `#if IS_MLA`, т.е. `HEAD_GRP_SIZE==128 && HEAD_ELEMS==576` (`trtllm/cpp/kernels/xqa/defines.h:36`) | sm_120, но только DeepSeek-MLA | **недостижимо при D=256** |
| MMHA (скалярный декод) | `trtllm/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h:790` — `dot(q,k)` в регистрах | все | тензорных ядер нет |
| flashMLA | `trtllm/cpp/tensorrt_llm/kernels/flashMLA/flash_fwd_mla_kernel.h:661` — `__CUDA_ARCH__ == 900` | sm_90 | нет |
| cascade (общий префикс, beam) | `trtllm/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cascadeAttentionKernel.h:36-45` | `>= 800` | не наш профиль |

Готовых кубинов под sm_120 в дереве **нет** (`trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/` — 86 архивов, ни одного `sm120`);
sm_120-ядра собираются из исходников в цель `_context_attention_kernels_120`
(`trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/CMakeLists.txt:97-117`).
XQA под sm_120 тоже без прекомпиляции — `trtllm/cpp/kernels/xqa/gen_cubins.py:37` перечисляет `arch_options = [80, 86, 90]`, остальное через NVRTC.

### 2. head_dim 256: где есть и что именно меняется

**Где есть.** `trtllm/cpp/kernels/fmha_v2/setup.py:5285` — `tiled_params_q_kv_step[256] = [64, 128]`
(q_step 64, kv_step 128) для всех sm с `sm_mma == 80`, включая 120
(`trtllm/cpp/kernels/fmha_v2/setup.py:5267`). fp8: `trtllm/cpp/kernels/fmha_v2/setup.py:6810-6815` —
`enumerate_qmma_flash_kernels(sm=120, dtype='e4m3_fp32', head_sizes=[64,128,192,256,576], output_dtype="bf16")`.
Рукописный мост: `trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_v2.cpp:270`
— `run_skip_softmax_bf16_d256_causal_sm120(...)`.

**Что меняется — 1: гранулярное разрезание контракции.**
`trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:58` — `CTA_P_TILE_K = D < 32 ? 16 : (D < 64 ? 32 : 64)`,
т.е. при D=256 BMM1 контрактирует голову **кусками по 64**, а не по 256.
`trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:64-66` — `CTA_O_TILE_K = DV > 256 ? 16 : (DV > 128 ? 32 : 64)`,
т.е. при DV=256 BMM2 контрактирует ключи **кусками по 32** (из kv-шага 128 — четыре куска),
и `CTA_O_TILE_N = DV > 256 ? 256 : DV` (строка 62).
Отсюда `RELOAD_Q = (CTA_P_TILE_K != D)` — `trtllm/cpp/kernels/fmha_v2/src/fmha/kernel_traits.h:246`;
при D=256 это **true**, и Q перечитывается из глобальной памяти на каждом kv-шаге
(`trtllm/cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_noloop_tiled.h:345-352`).

**Что меняется — 2: двойная буферизация становится возможной.**
`trtllm/cpp/kernels/fmha_v2/src/fmha/kernel_traits.h:398-408` — при `USE_GRANULAR_TILING`
у Q (при D>64), K и V по **2** smem-буфера. Арифметика для D=256, bf16, Br=64, Bc=128:
Q 64×64×2×2 = 16 KiB, K 128×64×2×2 = 32 KiB, V 32×256×2×2 = 32 KiB → **80 KiB** при kv-шаге 128
и полном ping-pong. Без гранулярности та же геометрия требовала бы 64×256×2 + 2·128×256×2 = 160 KiB.

**Что меняется — 3: разбиение D между CTA включается только при DV>256.**
`trtllm/cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_noloop_tiled.h:121-124` —
`ctas_per_o_row = div_up(Cta_tile_o::VALID_N, Cta_tile_o::N)`, `o_part = blockIdx.x % ctas_per_o_row`.
При DV=256 `N == VALID_N == 256` ⇒ `ctas_per_o_row == 1`. **Split-D по CTA при head_dim 256 у них нет** —
он появляется только на MLA-генерации (DV=512), и там каждая CTA пересчитывает весь BMM1.

**Что меняется — 4: прямой хёристик, что при D≥256 выигрывает именно гранулярное ядро.**
`trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.cpp:416-419`:
```
else if ((isSm8x || isSm120f) && mFixedParams.headSize < 256)
{
    // flash attention tiled kernel is faster on Ada and Ampere derivatives when head_size>=256
    mLaunchParams.granular_tiling = false;
}
```
То есть на Ada и потребительском Blackwell гранулярное ядро включают **ровно и только** при head_dim ≥ 256.

**Что меняется — 5: регистровое давление.** Для D=256, Br=64, 4 warps: `MMAS_M = 1`,
`VALID_MMAS_N = 256/8 = 32` (`trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:285`).
При f32-аккумуляторе (`Fragment_accumulator<Ampere_hmma_bf16_traits>`, 8 регистров на фрагмент)
это 32×8/2 = 128 регистров только под `acc_o`. При f16-аккумуляторе
(`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:942` — `Fragment<uint16_t, 8>`, 4 регистра) — **64**.
В нетайловом (`_nl`) ядре при head_size ≥ 256 они дополнительно включают
`limit_qk_fragments = limit_v_fragments = True` и уменьшают kv-шаг до 16
(`trtllm/cpp/kernels/fmha_v2/setup.py:5349-5360`), а LDGSTS для K и V выключают
(`trtllm/cpp/kernels/fmha_v2/setup.py:5336-5338`).

### 3. Контекстное ядро под sm_120 (разбор)

Умолчание для bf16 + causal + PACKED_QKV + D∈{128,256} — рукописный warp-spec TU;
всё остальное падает в кубин/лаунчер-путь: `trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_v2.cpp:279-303`.

- Конфигурация: `S=128` (kv-шаг), `VALID_D=VALID_DV=256`, `STEP_Q=64`, `WARPS_M=4`, `WARPS_N=1`,
  1 producer-warp — `trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/skip_softmax_sm120/fused_multihead_flash_attention_ws_sm120.cu:61-73`.
  Сетка `(q_tiles, H, B)`, блок 5 warps = 160 нитей (строки 129-131).
- Раскладка Q/K: 128-байтовые smem-строки, XOR-свизл `(col/8) ^ (row%8)` —
  `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md:74-79`. Это **побитово тот же свизл**,
  что в NInfer (`ninfer/src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh:78-80`).
- V переразбит на куски шириной 64 по DV, чтобы строка тоже стала 128-байтовой —
  `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/kernel_traits.h:138-143`; итого 16 подплиток V
  на один kv-шаг (`kernel_traits.h:289-292`).
- Порядок циклов: BMM1 по 4 кускам головы (`compute_sync_mma.h:219-243`), затем softmax,
  затем BMM2 по (4 dv-куска × 4 kv-куска) (`compute_sync_mma.h:348-377`).
- Онлайн-softmax: `reduce<Max_>` даёт максимум **текущей** плитки
  (`trtllm/cpp/kernels/fmha_v2/src/fmha/softmax.h:3673-3676` — `dst[mi] = elt_[mi][0]`, дальше сворачивание),
  а слияние с бегущим максимумом делает `Tile_o_normalizer::update`:
  `curr_max = fmax(prev_max, curr_max); alpha = expf(prev_max - curr_max)`
  (`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:1518-1523`).
- Стадии конвейера: глубина = числу гранулярных буферов (2), `mbarrier` на слот вместо
  `__syncthreads` (`trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/kernel_traits.h:272-274`, `337-352`).
- `setmaxnreg` на sm_120 отсутствует — `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/README.md:93-98`
  (совпадает с нашим брифом).
- Эпилог: `Smem_tile_o` алиасит начало smem (Q/K/V уже не нужны),
  свизл + межварповая редукция, `named_barrier` только по consumer-группе —
  `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/compute_sync_mma.h:405-429`.
- Что порт **не** выигрывает, их же словами: перекрытия MMA/softmax нет, потому что
  `mma.sync` блокирует нить до коммита результата (`README.md:109-115`).

### 4. Декодное ядро (XQA) под sm_120

- Сетка `dim3{nbSubSeqPerSeq, nbKHeads, batchSize}`, блок 256 нитей = 8 warps
  (`trtllm/cpp/kernels/xqa/mha.cu:2899-2901`); Q-голов в сетке нет.
- Вся группа Q одной KV-головы кладётся в измерение M и **дополняется до 16**:
  `warpTile = {64, roundUp(nbValidRows, 16U)}` — `trtllm/cpp/kernels/xqa/mha.cu:76`.
  При 16Q/2KV (группа 8) утилизация M = 50 %, при 24Q/4KV (группа 6) — 37.5 %.
  Обрезаются только загрузка Q (`mha.cu:1700-1703`) и запись (`mha.cu:955-960`), сами MMA — нет.
- split-KV: `nbSubSeqPerSeq = min(max(1, SMcount/(batch*nbKHeads)), div_up(maxSeqLen, 256))`
  (`trtllm/cpp/kernels/xqa/mha.cu:2808-2825`). Разбиение **чередующееся**, а не непрерывное:
  `seqIterInit = nbSkipLeadingTiles + idxSubSeqInSeq`, шаг `nbSubSeqPerSeq`
  (`mha.cu:1740-1742`, `1917`).
- Свёртка частичных — **без отдельного ядра**: последняя CTA выигрывает по семафору
  `atom.acq_rel.gpu.global.inc.u32` с `lastOld = nbSubSeq-1`, счётчик самосбрасывается
  (`trtllm/cpp/kernels/xqa/mha.cu:2616-2625`), `writesOutput = isLastCta` (`mha.cu:2630`),
  само слияние — `mha.cu:2636-2716`.
- Страничный кеш: таблица блоков в регистрах на warp-плитку (`mha.cu:1772-1786`),
  `kBAD_PAGE_INDEX = -1` (`trtllm/cpp/kernels/xqa/utils.cuh:45`) превращается в нулевую
  заливку через `nullptr` в `HeadPtr::operator+` (`trtllm/cpp/kernels/xqa/mhaUtils.cuh:90-97`).
  **TMA в `mha.cu` не используется вообще** — только `cp.async` (`ldgsts.cuh`).
- Типы KV: bf16/f16, int8, e4m3 (`trtllm/cpp/kernels/xqa/defines.h:81-84`).
  Масштаб — **один скаляр на весь тензор, общий для K и V** (`trtllm/cpp/kernels/xqa/mha.h:135-136`).
  K-масштаб вклеен в `qkScale` до первого GEMM (`mha.cu:1756-1757`), V-масштаб — в эпилог
  (`mha.cu:2509`, `2522`). Деквант — в регистрах после `ldmatrix`, перед MMA,
  аппаратной `cvt.rn.f16x2.e4m3x2` при `__CUDA_ARCH__ >= 890`
  (`trtllm/cpp/kernels/xqa/utils.cuh:815-822`, `845-858`); в smem лежат сырые коды.
- Тензорные ядра всегда: `mma.sync.m16n8k16 …f32.f16.f16.f32` / `…bf16.bf16.f32`
  (`trtllm/cpp/kernels/xqa/mma.cuh:38`, `:48`), **аккумулятор всюду f32**.
  fp8-MMA (`mma.cuh:58`) в `mha.cu` не инстанцируется:
  `static_assert(is_same_v<InputElem, half> || bf16, "not implemented")` — `mha.cu:1071`, `1164`.
- head_dim 256 поддержан (`trtllm/cpp/kernels/xqa/mha.h:32`); отличие от 128 —
  `gemm1WarpsPerGrp` 2→4 и `gemm1NbWarpGrps` 2→**1** (`mha.cu:79-81`), т.е. **разбиение по
  длине последовательности внутри gemm1 исчезает**, все четыре варпа уходят на измерение головы.
- Спекуляция: M становится плоским (черновой токен × Q-голова),
  `nbTokenBlocksPerGrp = div_up(qSeqLen*headGrpSize, rowsPerBlock)` (`mha.cu:2896-2897`);
  маска — упакованный битмап, и её консультируют **только последние `actualQSeqLen` столбцов**
  (`mha.cu:537-539`, `1737`); отвергнутое ставится в `-1e5F`, а не `-inf`
  (`mha.cu:544`, `trtllm/cpp/kernels/xqa/utils.cuh:44`).

### 5. RoPE на части измерений

Частичный RoPE поддержан тремя разными способами:
- **Единичный коэффициент** (v1-препроцессинг): нити вне ротируемой зоны всё равно считают,
  но с `coef = (1,0)` — `trtllm/cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h:239`, `:266`.
- **Предикация целого вектора** (v2): `valid_rotary_dim_idx = head_dim_idx < rotary_embedding_dim`
  — там же `:820-821`, `:971`, `:990`.
- **«Крутим голову, хвост копируем»** (XQA sm_90): `storeUnrotatedTailForKV` / `storeUnrotatedTailForQ`,
  `trtllm/cpp/kernels/xqa/mha_sm90.cu:3471-3517`, `ROPE_ELEMS` — `trtllm/cpp/kernels/xqa/defines.h:170-174`.

Где считается: **никогда** в эпилоге QKV-GEMM и **никогда** внутри fmha_v2
(грепа `rope|rotary` по `cpp/kernels/fmha_v2/` — ноль попаданий в коде ядер).
Дефолт префилла — отдельное ядро `applyBiasRopeUpdateKVCache`, которое **одновременно**
добавляет bias, применяет RoPE и **пишет K/V прямо в страничный кеш**
(`trtllm/cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h:339`,
запись кеша `:578-579` (v1) и `:1005-1013` (v2); вызов —
`trtllm/cpp/tensorrt_llm/common/attentionOp.cpp:1962`).
cos/sin: v2 читает предпосчитанную таблицу (`…_2_template.h:954-956`), v1 считает `cosf/sinf`
из `inv_freq` с кэшем по позиции (`:260-262`, `:522`).

Отдельно — **слияние QK-нормы с RoPE**: `trtllm/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu`.
Один warp на пару (токен, голова), `head_dim` — шаблонный параметр (64/128/256,
`fusedQKNormRopeKernel.cu:406-432`); читает QKV один раз (`:184`), RMS через `warpReduceSum`
(`:207-210`), частоты пересчитывает `exp2f`/`__sincosf` **без таблицы** (`:257`, `:280`),
партнёрский элемент для neox берёт `__shfl_xor_sync` вместо shared-memory-транспонирования
(`:287-294`), частичный RoPE — ветка `is_full_rope` (`:329`, `:339-348`), пишет один раз (`:351`).
KV-кеш это ядро не трогает.

### 6. Маскирование и чанкованный контекст

- Границы kv-цикла режут полностью замаскированные плитки **до** входа в цикл:
  `valid_seqlen = CAUSAL ? min(q_sequence_start + Cta_tile_p::M, actual_kv_seqlen)`,
  `kv_loop_end = div_up(valid_seqlen, N)*N`
  (`trtllm/cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel_noloop_tiled.h:181-185`).
- Маска применяется только с диагональной плитки:
  `kv_mask_loop_start = (q_sequence_start / N) * N`, `apply_mask = has_alibi || kv_loop >= kv_mask_loop_start`
  (там же `:172`, `:325`). Полностью «внутренние» плитки не платят ни одной инструкции маски.
- Чанкованный префилл: сдвиг диагонали задаётся как
  `q_sequence_start += actual_kv_seqlen - actual_q_seqlen` (там же `:141`) — bottom-right,
  то же соглашение, что у нас.
- Скользящее окно двигает и `kv_loop_start`, и границы маскирования (там же `:190-210`).
- Произвольная маска — предпосчитанный упакованный битмап (`MASK_VERSION == 6`,
  `trtllm/cpp/kernels/fmha_v2/src/fmha/mask.h:550`), строится ядром
  `trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.cu`.
- MTP-маска — отдельная ветка `Mask_dispatcher<..., IS_MTP=true>`
  (`trtllm/cpp/kernels/fmha_v2/src/fmha/mask.h:860`), которая делит строку на
  `num_grouped_heads` и восстанавливает индекс токена: `mtp_token_idx = get_row(mi,ii) / num_grouped_heads_`
  (`mask.h:425`), `seqlen_ -= actual_q_seqlen / num_grouped_heads + 1` (`mask.h:364`).
- **Отдельно: «уже посчитанная часть KV» как отдельный запуск.** Для MLA есть путь,
  где внимание считается по одному чанку KV, а объединение делает отдельное ядро по (max, sum):
  `merged.x = max(pre.x, curr.x); merged.y = pre.y*e^{pre.x-m} + curr.y*e^{curr.x-m}`
  — `trtllm/cpp/tensorrt_llm/kernels/mlaChunkedPrefill.cu:200-214`, интерфейс
  `trtllm/cpp/tensorrt_llm/kernels/mlaChunkedPrefill.cuh:24-33`.

### 7. Квантованные пути и точность аккумулятора — главный ответ на вопрос оператора

**Пониженный аккумулятор существует, отгружен и является умолчанием для fp16-входа —
и на BMM1, и на BMM2.**

- `Traits_o = Traits_p` на всех архитектурах, кроме Volta —
  `trtllm/cpp/kernels/fmha_v2/src/fmha/kernel_traits.h:73-77`, `:155-157`. То есть выбор
  аккумулятора один на оба произведения.
- `Ampere_hmma_fp16_traits` (он же `Ada_hmma_fp16_traits`, `kernel_traits.h:50-53`):
  `Traits<Ampere, u16, u16, u16, uint16_t, uint16_t>` — `trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:496`;
  инструкция — `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
  (`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:951`, `:959`), фрагмент `Fragment<uint16_t,8>` = **4 регистра**
  (`fragment.h:942`).
- Это умолчание: все безусловные перечисления идут с `dtype='fp16'`
  (`trtllm/cpp/kernels/fmha_v2/setup.py:6691`, `:6799`); `fp16_fp32` (f32-аккумулятор)
  добавляется только под `ENABLE_HMMA_FP32` (`setup.py:6819-6830`), и по умолчанию выключен
  (`trtllm/cpp/kernels/fmha_v2/README.md:16`). Рантайм: `force_fp32_acc` принудительно ставится
  только для BF16 и E4M3 — `trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.cpp:323-326`;
  флаг по умолчанию `false` (`trtllm/cpp/tensorrt_llm/common/attentionOp.h:538`).
- **Кубины подтверждают**: 27 архивов `*_fp16_*` (f16-акк) против 12 `*_fp16_fp32_*`
  в `trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/`.

**Как защищаются от переполнения — три разных механизма, ни один не является пред-масштабированием Q:**

1. **Насыщающий клип на P.** Сырой f16-аккумулятор QK умножается на `scale_bmm1`
   прямо в f16 и клипуется к максимальному конечному half:
   `satfinite_h2(hmul2(acc.reg(k), params_scale_bmm1_))`
   — `trtllm/cpp/kernels/fmha_v2/src/fmha/softmax.h:991-994`; сам клип — `min.xorsign.abs.f16x2`
   с константой `0x7bff7bff` (= ±65504), `trtllm/cpp/kernels/fmha_v2/src/fmha/utils.h:1175-1185`.
   Комментарий-обоснование: «Satfinite in case of overflow (due to fp16 accumulation)» —
   `trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec/epilogue.h:266`.
2. **Свежий локальный аккумулятор на каждую kv-плитку + нормировка O на каждом шаге.**
   `local_acc_o` обнуляется внутри kv-цикла, в него идут все MMA плитки, и затем
   `update_o` складывает: `acc_o = (alpha*acc_o + local_o) * beta`, где `beta = 1/curr_sum`
   — `trtllm/cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel.h:495-506` и
   `trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:2130-2211` (`beta` — строки `2148`, `2188`).
   Следствие: бегущий `acc_o` **всегда уже поделён на текущую сумму**, то есть является выпуклой
   комбинацией строк V и ограничен `max|V|` независимо от длины последовательности;
   `local_acc_o` ограничен `Bc·max|V|`.
3. **P для fp8 масштабируется фиксированной константой 256, вшитой в сдвиг максимума —
   ни одного лишнего умножения.**
   `Softmax_fp_quant_scale<e4m3_t>() = 256.f` (`trtllm/cpp/kernels/fmha_v2/src/fmha/numeric_types.h:53-59`,
   вывод: `2^floor(log2(448/1))`), связывается в трейтах
   (`trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:648-649`), применяется как
   `max_val = max[mi] - logf(SOFTMAX_FP_QUANT_SCALE)` в `apply_exp_with_mask`
   (`trtllm/cpp/kernels/fmha_v2/src/fmha/softmax.h:1530`), снимается в конце:
   `global_sum[mi] *= SOFTMAX_FP_DEQUANT_SCALE` (`trtllm/cpp/kernels/fmha_v2/src/fmha/warpspec/epilogue.h:1471`).
   Хостовый `scale_softmax` при этом равен 1.0 и «не используется»
   (`trtllm/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.cpp:237-238`).
   Для int8 масштаб зависит от длины: `scale_softmax = max(512.f, (float)s)`
   (`trtllm/cpp/kernels/fmha_v2/src/fused_multihead_attention.cpp:984-986`).

**fp8-путь (отгружен, включая sm_120 и D=256):** `Ada_qmma_e4m3_fp32_traits`
= `Traits<Ada, e4m3, e4m3, e4m3, float, float>` (`trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:635`),
инструкция `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
(`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:1155`). Так как `Traits_o == Traits_p`,
**оба произведения идут в e4m3**: P квантуется в e4m3, а **V лежит в smem прямо кодами**
(`Smem_tile_v<Ada_qmma_e4m3_fp32_traits>` наследует `Smem_tile_v_ampere_8bit_mma`,
`trtllm/cpp/kernels/fmha_v2/src/fmha/smem_tile_v.h:1361-1366`) — **никакого расширения V до f16 нет вообще**.
Гранулярное разбиение при этом принудительно отключается для e4m3 на sm_89/sm_120
(«Ada QMMA only supports non-tiled kernels») — `fmhaRunner.cpp:404-408`;
отгружаемая геометрия на D=256 видна в имени кубина
`fmha_v2_flash_attention_e4m3_fp32_64_32_S_q_paged_kv_256_sm89.cubin.tar.zst` (Br 64, Bc 32).

**fp8 + f16-аккумулятор существует в исходниках, но не собирается.**
`Ada_qmma_e4m3_fp16_traits` = `Traits<Ada, e4m3, e4m3, e4m3, uint16_t, uint16_t>`
(`trtllm/cpp/kernels/fmha_v2/src/fmha/traits.h:619`), инструкция
`mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` (`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:1190`),
с полным стеком smem/softmax/эпилога. Собирается только под `ENABLE_SM89_QMMA` и только для
не-flash v2-ядер (`setup.py:5482`, `:6741-6742`); все flash-перечисления идут `e4m3_fp32`
(`setup.py:6743-6758`, `:6811-6815`) с комментарием «fp8 or bf16 always accumulates on fp32»
(`setup.py:3465-3467`). Причина видна в коде: у `Ada_qmma_e4m3_fp16_traits`
**нет `SOFTMAX_FP_QUANT_SCALE`** (сравните `traits.h:619-631` и `traits.h:648-649`), поэтому
`Softmax_qmma` для него не переопределяет `apply_exp_with_mask`, и P ∈ [0,1] уходит в e4m3
без динамического диапазона.

**Жёсткое правило про bf16:** «BF16 MMA must accumulate with at least FP32» —
`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:971`, `:1053`; то же в рантайме,
`fmhaRunner.cpp:323`. Цена f16-аккумуляции у них зафиксирована эмпирически, а не оценкой:
допуск в тестах поднят с 0.015 (`trtllm/cpp/kernels/fmha_v2/src/fused_multihead_attention.cpp:538`)
до 0.02 с комментарием «NOTE: HALF_ACCUMULATION_FOR_FLASH_ATTENTION has larger epsilon»
(`trtllm/cpp/kernels/fmha_v2/test_sm80_configs.sh:49-57`).

---

## Что у нас сегодня

- **Префилл, bf16-кеш**: Br=64, Bc=64, D=256, 4 варпа/128 нитей, smem
  `(64 + 2·64)·256·2 = 98304 B = 96 KiB`, **одинарная буферизация**
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh:21-26`,
  `ninfer/src/ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh:1-11`).
  Голова целиком лежит в smem — гранулярного разбиения по D нет.
  PV: `mma_bf16` = `.f32.bf16.bf16.f32` (`prompt_bf16.cuh:392-395`,
  `ninfer/src/ops/common/mma.cuh:34-41`), P пакуется в **bf16** (`prompt_bf16.cuh:306-311`),
  аккумулятор `float acc[32][4]` = 128 регистров (`prompt_bf16.cuh:174-179`), **нормировка
  делается один раз в эпилоге** (`prompt_bf16.cuh:404-407`).
- **Префилл, int8-кеш**: Br=64, Bc=64, 16 варпов/512 нитей, smem 92672 B
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/prompt_i8.cuh:20-46`),
  `__maxnreg__(120)` (`prompt_i8.cuh:81`).
  Разбиение D **между варпами внутри CTA**: 4 row-tiles × 4 D-consumers, у каждого варпа
  `acc[8][4]` = 32 регистра (`prompt_i8.cuh:26-28`, `:417-422`).
  QK — `mma_s8` (int8, s32-акк). PV — `mma_f16` = `.f32.f16.f16.f32` из staging-тайла `v_f16`
  (`prompt_i8.cuh:437-440`); staging-тайл занимает `64·256·2 = 32768 B` из 92672 B арены
  (`prompt_i8.cuh:34-36`).
- **Префилл, fp8-кеш**: то же строение, smem 92416 B, `v_f16` те же 32768 B
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh:19-45`);
  QK — `mma_fp8_e4m3` = `mma.sync.aligned.kind::f8f6f4.m16n8k32…f32.e4m3.e4m3.f32`
  (`ninfer/src/ops/common/mma.cuh:60-67`), PV — `mma_f16`.
- **Свизл** — `(((col>>3) ^ (row&7))<<3) | (col&7)`
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/prompt_common.cuh:78-80`), тождественен их
  XOR-паттерну, но у нас строка 512 байт, у них 128.
- **Маскирование**: границы `n_block_max = (max_query_abs / Bc) + 1`, маска применяется только
  в неполных плитках через `full_score_tile` (`prompt_bf16.cuh:181-183`, `:255-258`) — то же,
  что у них, отдельного «пропуска полностью замаскированных плиток» не нужно.
- **RoPE**: отдельное ядро, in-place по Q и K, одна CTA на токен, `sincosf` из
  `__constant__`-таблицы обратных частот, смем-кеш cos/sin на токен
  (`ninfer/src/ops/kernel/rope.cuh:117-158`). Частичный RoPE честный: трогается ровно
  64 из 256 измерений (`rope.cuh:101-115`, `kHalf=32`). **Активны только 16 из 32 лейнов варпа**
  (`rope.cuh:106` — `if (lane >= kHalfPair) return;`, `kHalfPair = 16`).
- **Порядок операций на входе во внимание** (5 отдельных ядер):
  `rmsnorm(q)` → `rmsnorm(k)` → `rope` → `kv_cache_append` → `causal_softmax_attention`
  (`ninfer/src/targets/qwen3_6/impl/runtime/text_context_impl.h:841-842`, и в MTP-ветке
  `:382-385`, `:494-496`).
- **Декод, split-K**: частичные пишутся в `partial_acc/partial_m/partial_l`, свёртка —
  **отдельным ядром** `causal_attention_small_t_reduce_output_kernel`
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/small_t.cuh:157-161`,
  запуск — `ninfer/src/ops/softmax_attention/dense/causal_cache/small_t.cu:316-330`).
  Число сплитов — хёристик по окну (`small_t.cuh:81-119`), потолок `85·SmallTSplitScale`
  (`ninfer/src/ops/softmax_attention/dense/causal_cache/geometry.cuh:12`).
- **Геометрия голов**: `CausalD256H24Kv4`, `CausalD256H16Kv2` (`geometry.cuh:15-16`).

---

## Кандидаты для NInfer

### 1. f16-аккумулятор на PV + их схема защиты от переполнения

**Механизм.** В `prompt_i8.cuh` / `prompt_fp8.cuh` / `small_t_*` оба операнда PV **уже** f16
(`p_s` — `__half*`, `v_f16` — `__half*`), а аккумулятор f32: `mma_f16` даёт ярус 257.
Замена на `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` поднимает PV на ярус **506.6**
и вдвое сокращает регистры под `acc`. Сам по себе этот рычаг у нас уже замерен
(память `ninfer-attention-prefill-ceiling`: +4.3…9.2 % на int8-KV), но не отгружен —
блокер именно численный. TRT-LLM отгружает f16-аккумулятор на обоих произведениях уже
годы, и в коде лежит ровно то, чего у нас нет:
- бегущий `acc_o` держится **уже нормированным**: `acc_o = (alpha·acc_o + local_o)·(1/curr_sum)`
  (`trtllm/cpp/kernels/fmha_v2/src/fmha/fragment.h:2148`, `:2188`) ⇒ |acc_o| ≤ max|V| при любой длине;
- MMA пишет в **свежий** `local_acc_o`, обнуляемый на каждой kv-плитке
  (`trtllm/cpp/kernels/fmha_v2/src/fused_multihead_flash_attention_kernel.h:495-498`) ⇒
  |local_acc_o| ≤ Bc·max|V| = 64·max|V|, что при max|V| ≲ 100 не приближается к 65504;
- на стороне P — `satfinite_h2` (`softmax.h:991-994`, `utils.h:1175-1185`).

У нас сегодня `acc` **ненормирован до самого эпилога** (`prompt_i8.cuh:472-476`), то есть
растёт как `l·max|V|` — это и есть источник риска, который снимает их схема.

**Ожидаемый эффект.** Арифметика по ярусам на плитку (Br=64, Bc=64, D=256; QK и PV по
2·64·64·256 = 2.10 MFLOP каждый):

| путь | QK ярус | PV ярус | время выдачи MMA (отн.) | доля PV |
|---|---|---|---|---|
| int8-кеш сегодня | 1026.7 | 257 | 2.05 + 8.17 = 10.22 | 80 % |
| int8-кеш + f16-акк | 1026.7 | 506.6 | 2.05 + 4.14 = **6.19** | 67 % |
| fp8-кеш сегодня | 517.4 | 257 | 4.06 + 8.17 = 12.23 | 67 % |
| fp8-кеш + f16-акк | 517.4 | 506.6 | 4.06 + 4.14 = **8.20** | 50 % |

то есть −39 % (int8) и −33 % (fp8) времени выдачи MMA внутри ядра внимания.
Внимание — 28.5 % префилла на 33K и 44.2 % на 67K, значит потолок выигрыша по префиллу
11.1 % / 17.2 % при полной MMA-связанности; уже измеренные +4.3…9.2 % лежат внутри этого
конверта, что подтверждает модель. Плюс регистры: `acc[8][4]` → `acc[8][2]`, 32 → 16 регистров
на варп при `__maxnreg__(120)` — 13 % бюджета освобождается под более глубокий конвейер.
На bf16-путь (`prompt_bf16.cuh`) кандидат **не переносится**: там P пакуется в bf16, а V —
bf16 из кеша, и приведение V к f16 поплиточно — это ровно опровергнутый «маршрут B» (−37 %).

**Чем меряем.** A/B на `prompt_i8` и `prompt_fp8`, чанк 8192, mtp3, два прохода бенча:
(A) сегодняшний `mma_f16`; (B) `mma_f16_f16acc` + пошаговая нормировка `acc` на `1/l`
(fold `beta` в существующий цикл `alpha`) + свежий `local_acc` на плитку.
Точность — стенд A/B/C из `ninfer-accuracy-first-benchmarks`, AIME25 плюс поэлементная
норма ‖O_B − O_A‖∞/‖O_A‖∞ на 33K и 67K контексте (у NVIDIA цена зафиксирована как
подъём допуска 0.015 → 0.02, `test_sm80_configs.sh:49-57` — ждать побитового равенства нельзя).
Обязательно проверить край: строка, где вся масса на одном ключе, и строка с почти
равномерным вниманием на 67K (максимальное `l`).

**Риски и что ломается.** Вывод перестаёт быть побитовым — это заявка категории
«приближение», не «порядок». Пошаговая нормировка добавляет D=256 умножений на строку
на плитку (у нас уже есть такой же цикл под `alpha`, `prompt_i8.cuh:417-422`) — то есть
стоимость близка к нулю, но `1/l` придётся считать каждый шаг (`__frcp_rn`, 1 на строку).
Если `l` в первой плитке нулевой (полностью замаскированная строка) — нужна их же
защита `beta = (sum == 0 || sum != sum) ? 1.f : 1.f/sum` (`fragment.h:2147`, `:2186`).

**Оценка объёма.** `ninfer/src/ops/common/mma.cuh` (+1 функция), `prompt_i8.cuh`,
`prompt_fp8.cuh`, `small_t_i8.cuh`, `small_t_fp8.cuh` — 5 файлов.

---

### 2. Полностью fp8 PV: P квантуется фиксированным масштабом 256, V остаётся кодами

**Механизм.** У них при e4m3 `Traits_o == Traits_p`, поэтому PV идёт `e4m3 × e4m3 → f32`
и **V вообще не расширяется** — лежит в smem кодами (`smem_tile_v.h:1361-1366`).
Чтобы P влезло в e4m3 без потери диапазона, они не делают динамический max по плитке:
они сдвигают сам максимум софтмакса на `logf(256)` (`softmax.h:1530`), так что
`exp` сразу выдаёт `256·exp(x−max) ∈ [0, 256] < 448 = MAX_E4M3`, и снимают масштаб
одним умножением суммы в эпилоге (`warpspec/epilogue.h:1471`). Ноль дополнительных
умножений в горячем цикле. Наш случай отличается тем, что у нас **построчный** масштаб V
(`fp8_e4m3_row_codec.cuh`), а у них — один скаляр на тензор; построчный масштаб идёт по
измерению контракции, поэтому его нельзя вынести в эпилог — но можно вклеить в P:
`P'[i,j] = p[i,j]·(scale_v[j]/S)·256`, где `S = max_j scale_v[j]` по плитке; `S` уходит
в тот же пошаговый фолд, что и `alpha`.

**Ожидаемый эффект.** Три вещи сразу:
(а) PV с яруса 257 на 517.4 — время выдачи MMA fp8-ядра 12.23 → 8.12 (−34 %);
(б) исчезает staging-тайл `v_f16` — **32768 из 92416 байт smem** (`prompt_fp8.cuh:33-35`),
    чего хватает либо на двойную буферизацию K/V, либо на Bc=128
    (Q 16384 + K 32768 + V 32768 + P 16384 ≈ 98 KiB);
(в) исчезает вся работа по расширению V (`causal_prompt_fp8` widen + scale) —
    ровно та ALU-работа, которая в «маршруте B» стоила −37 %.
Комбинация с кандидатом 1 (`Ada_qmma_e4m3_fp16_traits`, ярус 1028.9) даёт
4.06 + 2.04 = 6.10 (−50 %), но см. риски.

**Чем меряем.** A/B на `prompt_fp8` при Bc=64 (чтобы изолировать эффект яруса), затем
второй A/B на Bc=128 с освободившейся памятью. Точность: сначала оффлайн — посчитать
на реальных активациях 33K распределение `scale_v[j]/S` внутри плитки Bc=64; если разброс
больше ~2^4, e4m3 (3 бита мантиссы) съест младшие строки и заявку надо закрывать до
всякой реализации. Только после этого — AIME25 на стенде A.

**Риски и что ломается.** Главный — точность: e4m3 у P даёт относительную ошибку ~6 % на
элемент против ~0.05 % у f16; усреднение по Bc=64 слагаемым даёт ~0.8 % на O.
NVIDIA отгружает это только для моделей, которые целиком в fp8. Для 35B (веса bf16,
кеш int8) это новая аппроксимация; для 27B NVFP4 — естественнее.
Второй риск: `mma_fp8_e4m3` у нас записан как `kind::f8f6f4` (`ninfer/src/ops/common/mma.cuh:63`),
а у них — простая форма без `kind::` (`fragment.h:1155`); нужен ISA-пробник, какая из форм
принимает `.f16`-аккумулятор на sm_120a (бриф говорит, что **блочно-масштабированные**
`mxf4nvf4/mxf4/mxf8f6f4` принимают только f32 — `kind::f8f6f4` в этот список не входит,
а таблица ярусов даёт измеренные 1028.9 для «fp8 e4m3 f16acc», значит какая-то форма работает).

**Оценка объёма.** `prompt_fp8.cuh`, `small_t_fp8.cuh`, `mma.cuh`, возможно
`fp8_e4m3_row_codec.cuh` — 4 файла. Если менять гранулярность масштаба V с построчной на
поплиточную — плюс `kv_cache/append`.

---

### 3. Гранулярное разбиение головы по D: Bc 64 → 128 и двойная буферизация

**Механизм.** Не держать всю голову 256 в smem. BMM1 контрактирует по кускам 64
(`traits.h:58`), BMM2 — по кускам 32 ключа (`traits.h:66`), каждый кусок в 2 буферах
(`kernel_traits.h:398-408`). Это переводит арену с «96 KiB, одинарная буферизация, Bc=64»
на «80 KiB, двойная буферизация, Bc=128» при том же head_dim 256.
Их собственный хёристик прямо говорит, что на Ada/Blackwell-consumer это выигрывает
**именно при head_dim ≥ 256** (`fmhaRunner.cpp:416-419`).

**Ожидаемый эффект.** Bc 64 → 128 вдвое сокращает число итераций kv-цикла и, значит,
число построчных редукций max/sum, пересчётов `alpha`, и `__syncthreads` на плитку;
на 33K это 516 → 258 итераций на CTA. Прямой выигрыш по MMA — нулевой (тот же объём
FLOP на том же ярусе); выигрыш — в накладных на плитку и в перекрытии загрузки с MMA,
которое сейчас одностороннее (одна арена, `cp_wait<0>` + `__syncthreads` дважды за плитку,
`prompt_bf16.cuh:196-197`, `:353-354`).

**Чем меряем.** Прямой A/B на `prompt_bf16` (там нет квантования, эффект чище):
(A) сегодня; (B) гранулярная арена с Bc=128. Метрика — TTFT прогретого 33K и 67K, и
доля ядра внимания в профиле. Обязательно замерить и обратную сторону — трафик перечитывания Q.

**Риски и что ломается.** У них цена схемы — `RELOAD_Q = true` (`kernel_traits.h:246`):
Q-плитка 64×256×2 = 32 KiB перечитывается на **каждом** kv-шаге. При Bc=128 и 33K это
258 перечитываний × 32 KiB × число CTA. Это тот же класс ошибки, что уже стоил нам −37 %
на «маршруте B» (переработка на каждую CTA), но здесь это L2-резидентное чтение, а не ALU.
Гибрид «держать Q целиком (32 KiB, один буфер), чанковать только K» даёт
32 + 32 + 32 = 96 KiB и убирает перечитывание Q — этот вариант надо мерить первым.
Второй риск: `Bc=128` удваивает `acc_p` (16 → 32 фрагмента), а при 4 варпах это +64 регистра.

**Оценка объёма.** Переписывание тела `prompt_bf16.cuh` (staging + адресация ldmatrix) —
1-2 файла, но это самая тяжёлая из заявок по объёму работы.

---

### 4. Слить QK-норму, RoPE и запись KV-кеша в одно ядро

**Механизм.** У них на входе во внимание **одно** ядро: `applyBiasRopeUpdateKVCache` читает
QKV один раз, применяет RoPE и пишет K/V прямо в страничный кеш
(`unfusedAttentionKernels_2_template.h:339`, запись `:1005-1013`), а для моделей с QK-нормой
есть `fusedQKNormRopeKernel.cu` — один warp на пару (токен, голова), RMS через
`warpReduceSum` (`:207-210`), частоты `exp2f`/`__sincosf` без таблицы (`:257`, `:280`),
частичный RoPE веткой `is_full_rope` (`:329`, `:339-348`), одна запись (`:351`).
У нас это пять ядер (`text_context_impl.h:841-842`, `:382-385`).

**Ожидаемый эффект.** Трафик HBM на токен на слой (35B, 16Q/2KV, D=256, bf16):
сегодня Q-цепочка `rmsnorm` (8192 r + 8192 w) + `rope` (2048 r + 2048 w) = 20480 B;
слитая — 8192 r + 8192 w = 16384 B. С K — экономия ≈ 4608 B/токен/слой.
На чанк 8192 × 10 GQA-слоёв = 377 MB, то есть 0.21 мс при 1792 GB/s (21 % пикового HBM
на время этих ядер, но **~0.1 % полного времени префилла** — префилл считающе-связан).
**Реальный адресат — декод**: там убираются 2 узла графа на GQA-слой (20 на шаг),
каждый с полом device-side launch ~2-3 мкс; при шаге 1.73 мс (579 tok/s) это до ~2 %.
Плюс в ядре RoPE сейчас простаивает половина варпа (`rope.cuh:106`).

**Чем меряем.** Декод: 35B, mtp3, замер tok/s до/после; счётчик числа узлов в графе.
Префилл: TTFT прогретого 33K. Вывод должен остаться **побитовым**, если порядок
арифметики RMS сохранён — это заявка категории «порядок», её надо проверять
поэлементным сравнением, а не только бенчем.

**Риски и что ломается.** RMS-редукция по 256 элементам сейчас делается ядром `rmsnorm`
с его порядком суммирования; warp-редукция по 32 нитям (по 8 элементов на нить) даст
другой порядок ⇒ вывод не побитовый. Если нужна побитовость — надо воспроизвести
существующий порядок. Слияние с `kv_cache_append` дополнительно смешивает квантование
(построчный absmax по K) с RoPE — absmax должен считаться **после** RoPE.

**Оценка объёма.** Новое ядро + удаление трёх вызовов: `ops/kernel/` (+1 файл),
`ops/launcher/`, `ops/wrapper/`, `text_context_impl.h` — 4-5 файлов.

---

### 5. Свёртка split-K последней CTA вместо отдельного ядра (декод)

**Механизм.** Вместо отдельного reduce-ядра каждая CTA после записи своих частичных
инкрементирует семафор `atom.acq_rel.gpu.global.inc.u32` с потолком `nbSubSeq-1`;
счётчик **самосбрасывается** в 0 (wrap на `.inc`), поэтому обнулять его между вызовами
не нужно; CTA, увидевшая `old == lastOld`, делает слияние сама
(`trtllm/cpp/kernels/xqa/mha.cu:2616-2630`, слияние `:2636-2716`).
Дополнительно у них разбиение по KV **чередующееся**, а не непрерывное
(`mha.cu:1740-1742`) — при каузальной маске это выравнивает нагрузку между сплитами.

**Ожидаемый эффект.** Трафик частичных не меняется (он всё равно идёт через глобальную
память): при 16 Q-голов × D=256 × bf16 × 85 сплитов это 696 KB на слой, запись+чтение
1.39 MB, ×10 GQA-слоёв = 13.9 MB на шаг → 7.8 мкс при 1792 GB/s (0.45 % шага 1.73 мс).
Убирается **не трафик, а запуск и сток сетки**: 10 узлов графа на шаг.
Отдельно стоит проверить чередующееся разбиение: сейчас `causal_small_t_*_splits`
режет окно на непрерывные куски, и при каузальной маске последние сплиты тяжелее.

**Чем меряем.** Декод 35B/27B, mtp0 и mtp3, tok/s. Два варианта отдельно:
(B1) только чередующееся разбиение (вывод меняется — меняется порядок суммирования частичных);
(B2) B1 + слияние в последней CTA.

**Риски и что ломается.** У них слияние делает **одна** CTA на (запрос, KV-голова), тогда
как наше reduce-ядро распараллелено по `(QHeads, D/kDChunk, …)` (`small_t.cu:316-317`) —
при 85-170 сплитах и D=256 однократная свёртка может оказаться **дороже** сэкономленного
запуска. Это заявка с реальным шансом уйти в минус; мерить обязательно до реализации,
достаточно прикинуть время свёртки одной CTA. Порядок суммирования частичных меняется ⇒
вывод не побитовый.

**Оценка объёма.** `small_t.cuh`, `small_t.cu`, `small_t_i8.cuh`, `small_t_fp8.cuh` — 4 файла.

---

## Опровергнуто / не переносится

- **warp-специализация sm_120 + TMA (`warpspec_sm120/`).** Читал целиком; выигрыш, который
  они сами заявляют, — «меньше инструкций загрузки» и «ожидание по слоту вместо
  `__syncthreads`» (`README.md:102-108`), а перекрытия MMA/softmax нет по признанию авторов
  («there is no `wgmma.async` on sm_120», `README.md:111-114`). У нас и warp-специализация,
  и TMA для страничного KV уже опровергнуты замером. Плюс `setmaxnreg` на карте нет —
  их же слова (`README.md:93-98`).
- **`trtllmGenKernels/fmha`.** Отсекается одной строкой: `mUseTllmGen = isSM100Family()`
  (`fmhaDispatcher.cpp:55`). На sm_120 недостижимо целиком.
- **skip-softmax (пропуск kv-плитки по голосованию варпа).** Механизм: `skip = (tile_max −
  running_max) < log(threshold/L)` по всем строкам нити, затем `__all_sync`
  (`warpspec_sm120/compute_sync_mma.h:288-294`), при пропуске восстанавливается прежний max
  (`:300-305`) и BMM2 не выполняется (`:346`). Работает и на head_dim 256 — то есть наш
  прежний аргумент закрытия («их путь промоушена есть только для 64/128») здесь **не
  применим**, и это надо честно зафиксировать. Но: (а) это аппроксимация, порог не
  универсальная константа, а **калиброванная по чекпоинту** формула numexpr от целевой
  разреженности (`trtllm/tensorrt_llm/_torch/attention_backend/sparse/skip_softmax/params.py:116-156`);
  (б) в warp-spec варианте плитки V всё равно вычитываются, чтобы не сорвать конвейер
  (`compute_sync_mma.h:341-343`), то есть экономятся только MMA, не полоса;
  (в) продуктовый адресат — DiT/видео (`trtllm/tensorrt_llm/visual_gen/sparse_attention.py:53`),
  не LLM-префилл. Для accuracy-first движка это заявка того же класса, что уже закрытая
  динамическая разреженность, и без калибровочного стенда она беспредметна.
- **XQA-раскладка «вся Q-группа в измерение M».** При наших группах 8 и 6 их
  `roundUp(nbValidRows, 16)` (`mha.cu:76`) даёт 50 % и 37.5 % полезной работы MMA.
  У нас уже CTA-на-KV-голову и на декоде тензорные ядра **измеренно проигрывают**
  (118.4 против 119.0 мкс) — заимствовать нечего.
- **XQA-обработка масштабов KV** (один скаляр на тензор, вклеенный в `qkScale` и в эпилог,
  `mha.cu:1756-1757`, `:2509`). Не переносится: у нас построчные (fp8) и погрупповые G64
  (int8) масштабы, они лежат вдоль измерения контракции и в эпилог не выносятся.
  Наш выбор точнее — см. память `ninfer-int8-vs-fp8-activations`.
- **`mla_sm120.cu`.** Единственный fp8-MMA-путь под sm_120 в XQA, но заперт за
  `#if IS_MLA` = `HEAD_GRP_SIZE==128 && HEAD_ELEMS==576` (`defines.h:36`) — при D=256
  недостижим. Плюс он использует `__cluster_dims__` (`mla_sm120.cu:1801`) и барьеры под
  `__CUDA_ARCH__ >= 900`.
- **flashMLA, MMHA, cascade.** flashMLA — `__CUDA_ARCH__ == 900`
  (`flash_fwd_mla_kernel.h:661`). MMHA — скалярные dot-произведения без тензорных ядер
  (`decoderMaskedMultiheadAttentionTemplate.h:790`), заведомо медленнее нашего декода.
  Cascade — общий префикс для батча с beam-поиском (`cascadeAttentionKernel.h:36-45`),
  не наш профиль (concurrency 1..8, без beam).
- **Split-D по CTA при head_dim 256.** Проверил специально: `ctas_per_o_row` при DV=256
  равен 1 (`noloop_tiled.h:121-124` в связке с `traits.h:62`). Механизм включается только
  при DV=512 и там каждая CTA **пересчитывает весь BMM1** — прямая потеря половины работы.
  Наше разбиение D между варпами внутри CTA (`prompt_i8.cuh:26-28`) строго лучше.
- **Слияние чанков префилла через отдельное (max,sum)-ядро** (`mlaChunkedPrefill.cu:200-214`).
  Нужно, когда KV одного чанка не помещается или считается отдельным запуском. У нас
  история уже лежит в страничном кеше и читается одним ядром; на чанке 8192 и 16 головах
  это 2048 CTA — параллелизма достаточно, дробить kv-цикл незачем.
- **`Ada_qmma_e4m3_fp16_traits` «как есть».** Полный стек в исходниках
  (`traits.h:619`, `fragment.h:1190`, `smem_tile_v.h:1378`, `softmax.h:3299`), но не собирается,
  и причина видна: у трейта **нет** `SOFTMAX_FP_QUANT_SCALE`, поэтому P ∈ [0,1] попадает в
  e4m3 без диапазона. Копировать нельзя — надо доносить масштаб 256 самим (см. кандидат 2).

---

## Открытые вопросы

1. **Какая форма fp8-MMA на sm_120a принимает `.f16`-аккумулятор.** У нас в `mma.cuh:63`
   стоит `kind::f8f6f4`; у них — простая `m16n8k32` без `kind::` (`fragment.h:1155`, `:1190`).
   Таблица ярусов брифа даёт измеренные 1028.9 для «fp8 e4m3 f16acc», но не говорит, какой
   спеллинг проверялся. Нужен ISA-пробник по обеим формам.
2. **Влияет ли `kind::f8f6f4` на пропускную способность против простой формы при том же
   f32-аккумуляторе.** Если да — это отдельный бесплатный рычаг на QK в `prompt_fp8`.
3. **Реальная статистика `scale_v[j]` внутри плитки Bc=64** на наших активациях. Без неё
   кандидат 2 не оценить: если разброс построчных масштабов внутри плитки больше ~2^4,
   e4m3-квантование P съест младшие ключи, и заявку надо закрывать до реализации.
4. **Сколько именно стоит `RELOAD_Q`.** TRT-LLM платит перечитыванием Q-плитки на каждом
   kv-шаге и не комментирует цену нигде в коде. Прежде чем строить кандидат 3, надо
   померить это отдельным микротестом (L2-резидентное чтение 32 KiB × число kv-шагов).
5. **Что именно даёт двойная буферизация на нашей карте.** Бриф фиксирует, что глубокие
   асинхронные конвейеры регрессируют на +9…15 %, но там речь о глубине > 2. Двухстадийный
   ping-pong (их `BUFFERS_PER_TILE = 2`) — это другая точка, и она не измерялась.
6. **Порог, при котором пошаговая нормировка O становится дороже выигрыша от f16-акк.**
   При Bc=64 и D=256 это 256 умножений на строку на плитку поверх уже существующего цикла
   `alpha`; при Bc=128 амортизируется вдвое лучше. Кандидаты 1 и 3 связаны.
7. **Что TRT-LLM делает с `attention_sinks` при f16-акк** (`fragment.h:1814-1818` правит
   сумму на `SOFTMAX_FP_DEQUANT_SCALE`). У нас sinks нет, но если появятся — там есть
   готовая ловушка.
