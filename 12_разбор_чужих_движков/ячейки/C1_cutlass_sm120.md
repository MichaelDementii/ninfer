# C1 — CUTLASS на sm_120: block-scaled FP4 GEMM, и что в нём сломано

Клон CUTLASS 4.8.0 (`dc45f97`, релиз 2026-08-25). NInfer — `da49c0d`.

## Что это за механизм

CUTLASS держит для `arch::Sm120` отдельное, почти не пересекающееся с Hopper/B200 семейство:
пять коллективов mainloop (плотный, плотный ptr-array, block-scaled, block-scaled ptr-array,
2:4-разреженный ×2), один свой kernel-файл и свой эпилог. Все они — **warp-specialized
rmem-source**: TMA-продюсер грузит A/B/SFA/SFB в shared, потребитель тянет фрагменты
`ldmatrix`/`LDS` в регистры и зовёт обычный warp-level `mma.sync`. Никаких дескрипторов
операндов в shared (wgmma) и никакого tcgen05 — коллективы это прямо запрещают.
Ключ ко всему block-scaled — **предсвизленная на хосте раскладка масштабов**: блок 128 строк ×
4 масштаба = 512 байт, который TMA переносит как есть, а потребитель читает одним 4-байтовым
`LDS` на MMA. Мы делаем то же самое инструкцией и по той же схеме варпов (4×2), но с вдвое
более узкой плиткой по K, с K-мажорной раскладкой масштабов активаций и без регистровой
конвейеризации между K-шагами.

Отдельно: grouped block-scaled FP4 на SM120 у CUTLASS реализован через **девайс-сайд
переписывание TMA-дескрипторов** (`tensormap.replace`), и именно там живёт баг #3096.

## Доказательства из кода

### Карта: что вообще есть для sm_120 (вопрос 1)

- `cutlass/include/cute/arch/mma_sm120.hpp:1765-3278` — атомы MMA: `SM120_16x8x32_TN`
  (kind::f8f6f4, все пары из {e2m1,e3m2,e2m3,e4m3,e5m2}), `SM120_16x8x32_TN_VS`
  (mxf8f6f4 block-scaled) и `SM120_16x8x64_TN_VS` (mxf4nvf4 block-scaled).
- `cutlass/include/cute/arch/mma_sm120.hpp:3377-3421` — ровно наша инструкция:
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`.
  Сравнить с `ninfer/src/ops/common/mma.cuh:89-101` — совпадает пословно, включая
  `bid=0, tid=0`.
- `cutlass/include/cute/arch/mma_sm120_sparse.hpp:46-3421` — разреженное 2:4 семейство:
  `SM120_SPARSE_16x8x64_TN` / `_TN_VS` и `SM120_SPARSE_16x8x128_TN_VS` (наш ярус 3986.2 TOPS).
- `cutlass/include/cute/atom/mma_traits_sm120.hpp:135-200` — traits с раскладками SFA/SFB.
- Коллективы: `cutlass/include/cutlass/gemm/collective/sm120_{mma_tma, mma_array_tma,
  blockscaled_mma_tma, blockscaled_mma_array_tma, blockscaled_sparse_mma_tma, sparse_mma_tma,
  mma_tma_blockwise_scaling, mma_array_tma_blockwise_scaling}.hpp`.
- Билдеры: `cutlass/include/cutlass/gemm/collective/builders/sm120_{common, mma_builder,
  array_mma_builder, blockscaled_mma_builder, blockscaled_sparse_mma_builder,
  sparse_mma_builder, blockwise_mma_builder}.inl`.
- Единственный собственный kernel-файл:
  `cutlass/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`
  (только для разреженного пути). Всё остальное — переиспользованные sm90-ядра
  `sm90_gemm_tma_warpspecialized_cooperative.hpp` / `..._pingpong.hpp` /
  `sm90_gemm_array_tma_warpspecialized_cooperative.hpp` с ветками
  `IsSm120Family` (`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:191, 975-985`).
- Эпилог: `cutlass/include/cutlass/epilogue/collective/builders/sm120_builder.inl`,
  `sm120_common.inl`, `epilogue/fusion/sm120_{callbacks,visitor_store}_tma_warpspecialized.hpp`.
- Тесты: `cutlass/test/unit/gemm/device/sm120_{tensorop,blockscaled_tensorop,
  sparse_tensorop,blockscaled_sparse_tensorop}_gemm/` — 44 файла.
- `cutlass/include/cutlass/arch/arch.h:47` — `sm120_smem_capacity_bytes = 101376`
  (наши 99 КиБ).

### Разбор block-scaled FP4 mainloop (вопрос 2)

Файл `cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`.

- **Плитки.** `cutlass/python/cutlass_library/generator.py:11394-11416` — NVIDIA поставляет
  для nvf4/mxf4 sm120 ровно: `M∈{128,256}`, `N∈{8,16,32,64,128}`, `K∈{128,256}`, cluster 1×1×1.
  В юнит-тестах канон — `Shape<_128,_128,_256>`
  (`test/.../sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu:80`).
- **Атом MMA внутри плитки.** `builders/sm120_blockscaled_mma_builder.inl:116-118` —
  `PermTileK = _64` для nvf4 (для mxf8f6f4 — `_32`); `:131-134` — `AtomLayoutMNK`:
  cooperative и N≥16 → `Shape<_4,_2,_1>` (4 варпа по M, 2 по N — **то же, что у нас**),
  cooperative и N<16 → `Shape<_8,_1,_1>`, pingpong → `Shape<_2,_2,_1>`.
- **Стадии.** `:257-264` — из `sm120_smem_capacity_bytes` вычитается память планировщика/
  тензормапов, остаток делится на байты стадии. Для 128×128×256 nvf4 стадия =
  16384 (A) + 16384 (B) + 2048 (SFA) + 2048 (SFB) = 36864 Б → **2 стадии**, то есть 512
  элементов K в полёте. У нас 4 стадии × K=128 — те же 512 K и почти те же 77.8 КиБ.
- **Чем грузят.** `:296-320` — четыре TMA-дескриптора `SM90_TMA_LOAD`
  (`cp.async.bulk.tensor`), `_1{}` мультикаст («No programmatic multicast»); `:677-681` —
  все четыре `copy()` вешаются на **один** `tma_barrier`, транзакция считается суммой
  (`:256-264`). Это ровно наша схема (`ninfer/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:218-236`).
- **Свизл.** `builders/sm120_common.inl:125-135` — выбирается самый широкий атом, который
  делит `TileK`: `Layout_K_SW128_Atom` = `Swizzle<3,4,3>` над `Shape<_8,_1024>` бит =
  **128 байт на строку** (`cute/atom/mma_traits_sm90_gmma.hpp:84`). Для fp4 это K=256.
  При `TileK=128` (наш случай) остаётся `SW64` = `Swizzle<2,4,3>`, 64 Б/строку.
- **Порядок обхода K и конвейер регистров.** `:827-901`:
  `K_BLOCK_MAX = TileK/PermTileK` (для 256/64 = 4). Тело:
  `copy_kblock(_0{})` до цикла, дальше `copy_kblock(k_block_next); gemm_kblock(k_block);` —
  **фрагменты следующего K-шага грузятся до MMA текущего**. На последнем K-шаге стадии
  `consumer_release` + `++smem_pipe_read` + `consumer_wait` вставлены **перед** `gemm_kblock`,
  то есть ожидание mbarrier перекрыто MMA, чьи фрагменты уже в регистрах.
- **Эпилог.** `epilogue/collective/builders/sm120_builder.inl:191-197` — подплитка
  `EpiN = min(CTA_N,32)`, `EpiM = (CTA_N<16 ? 128 : 64)`; `:94-104` — `ReuseSmem`, когда
  `sizeof(C)==sizeof(D)>8` бит, `DelayTmaStore` когда C пустой, `StagesD=2`.
  Аккумуляторы уходят в smem подплитками и складываются TMA-store, перекрываясь с хвостом
  mainloop.

### Раскладка масштабов SFA/SFB (вопрос 3)

- `cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp:51-56` — атом:
  `Blk_MN = 128`, `Blk_SF = 4`,
  `SfKMajorAtom = Layout<Shape<Shape<_32,_4>, Shape<Int<SFVecSize>,_4>>,
  Stride<Stride<_16,_4>, Stride<_0,_1>>>`.
  Читается так: байтовое смещение = `16·(m mod 32) + 4·(m div 32) + k_scale`.
  Один неделимый блок — 512 байт = 128 строк × 4 масштаба; 4 подряд идущих байта — это
  4 масштаба **одной** строки. Комментарий в билдере
  (`builders/sm120_blockscaled_mma_builder.inl:186-189`) объясняет выбор: «чтобы
  последовательные 32 бита содержали масштабы ровно одной строки».
- `sm100_blockscaled_layout.hpp:86-114` — `tile_atom_to_shape_SFA/SFB` = `tile_to_shape` этого
  атома на `(M,K)` / `(N,K)`. Требование: **M (и N) кратно 128, K кратно 4·SFVecSize**
  (для NVFP4 VS=16 → K кратно 64).
- `builders/sm120_blockscaled_mma_builder.inl:194-216` — тот же атом разворачивается в
  smem-layout; `:179` — читается он тривиально:
  `SmemCopyAtomSF = Copy_Atom<UniversalCopy<uint8_t>, uint8_t>` с комментарием
  «auto-vectorized LDS». Никакого `ldmatrix`, никаких шафлов.
- `sm120_blockscaled_mma_tma.hpp:309-320, 370-381` — TMA для SF строится как
  `make_tma_copy<uint16_t>(...)`, хотя `ElementSF` однобайтовый. Причина в
  `cute/atom/copy_traits_sm90_tma.hpp:797`: внутренняя размерность TMA-бокса ограничена
  **256 элементами в единицах `TmaInternalType`**; взяв uint16 вместо uint8, они получают бокс
  512 байт вместо 256 — ровно один SF-блок.
- `cute/atom/mma_traits_sm120.hpp:158-162` — раскладка, которую требует само железо:
  SFA `(T32,V64)->(M16,K64)`, `Stride<Stride<_8,_0,_1>,_16>` → лейн `l` отвечает за строку
  `m = 8·(l&1) + (l>>2)`, режим `2:0` означает, что реально различных лейнов 16 (пары
  широковещательные). SFB: `Stride<Stride<_0,_1>,_8>` → `n = l>>2`, 8 эффективных лейнов.
  Сравнить: `ninfer/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:257-258` —
  `sfa_row = ((lane&1)<<3) | (lane>>2)`, `sfb_row = lane>>2`. **Идентично** — это диктует ISA,
  свободы тут нет.
- `sm120_blockscaled_mma_tma.hpp:203-214, 735-740` — при `TileShape_N < 128` SFB всё равно
  грузится блоками по 128 строк (`TileShapeSFB = max(N,128)`), а CTA берёт свою четверть
  индексом `n % (128/TileN)`. Гранулярность 128 — не оптимизация, а плата за формат.
- **Побочная находка**: `cute/atom/mma_traits_sm120.hpp:210-231` — `fp4_shift_A/B`. Когда fp4
  скармливают инструкции `kind::f8f6f4` (8-битные контейнеры), `ldmatrix .b4x16_p64` кладёт
  тетраду в младшие 4 бита, а MMA ждёт её в середине байта — нужен `v << 2` на каждый
  регистр. Для нашего пути `mxf4nvf4` (k64) это не нужно, и CUTLASS его туда не ставит.

### Grouped GEMM и баг #3096 (вопрос 4)

- `sm120_blockscaled_mma_array_tma.hpp:277-282` — 4 `cute::TmaDescriptor` в shared;
  `:456-461` — воркспейс `4 × sizeof(TmaDescriptor) × sm_count` в глобальной памяти;
  `:1036-1049` — подмена адреса, `:1053-1120` — подмена размеров и страйдов, включая
  `mainloop_params.layout_SFA[next_group]` (per-group хостовая раскладка SF);
  `:1145-1160` — `cp_fence_release` / `fence_acquire`.
- Планировщик — `sm90_tile_scheduler_group.hpp:314-395`: линейный индекс плитки
  раскладывается по экспертам окном в 32 группы, лейн-параллельным префикс-суммой
  (`__shfl_up_sync`) и `__ballot_sync`, с кэшем на 2 формы вперёд. Сетка считается на хосте
  из суммы плиток всех групп (`:215-240`), либо, если хостовых форм нет, просто
  `hw_info.sm_count`.
- **Где сломано.** Следов #3096 и обходов в дереве 4.8 нет (`grep 3096` — пусто; в CHANGELOG
  по SM120 правились другие вещи: `CHANGELOG.md:350-351, 540-541`). Ушёл в веб:
  [NVIDIA/cutlass#3096](https://github.com/NVIDIA/cutlass/issues/3096), «SM120 (Bug) (With Fix)
  (RTX Blackwell) NVFP4 MoE: CUTLASS Grouped GEMM Produces Garbage Output», автор
  brandonmmusic-max, 2026-03-09, **открыт**, официального ответа NVIDIA в треде нет. Симптом:
  плотный FP4 GEMM считает правильно, grouped FP4 на экспертах MoE выдаёт неверные числа
  (модель отвечает мусором). Затронуты 4.2.1 и 4.4.1, карта RTX PRO 6000 Blackwell (SM 12.0).
  Автор списывает это на сборку под `compute_120a` вместо `compute_120f` и обходит переходом
  на Marlin (W4A16) либо патчами FlashInfer.
- **Что говорит код** (первичный источник, и он объяснение автора не подтверждает):
  `cutlass/include/cutlass/arch/config.h:62-69` — `tensormap.replace` включается макросом
  `CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED`, а тот требует
  `__CUDA_ARCH_FEAT_SM120_ALL`, то есть именно **`-arch=sm_120a`**; в списке нет ни SM120F,
  ни plain SM120. Без этого макроса весь блок подмены дескрипторов
  (`cute/arch/copy_sm90_desc.hpp:343-355, 358-414`) вырождается в
  `CUTE_INVALID_CONTROL_PATH`, который на девайсе — `assert(0) + printf + __brkpt`
  (`cute/config.hpp:160`), то есть в release-сборке с `NDEBUG` это тихий no-op. Тихий no-op
  здесь = все группы читают тензормап группы 0 = мусор ровно того вида, что описан в issue.
  Направление у автора issue, похоже, перепутано: `120f` как раз **теряет**
  `CUTE_ARCH_LDSM_SM100A_ENABLED` (`cute/arch/config.hpp:131-137` — он включается только для
  `SM120A`/`SM121A`), а `tensormap.replace` там тоже не включён.
- **Второе, найденное чтением**: `cute/arch/copy_sm90_desc.hpp:366-370` —
  ```
  uint64_t const smem_int64_desc = 0;
  asm volatile ("cvt.u64.u32 %0, %1;" :: "l"(smem_int64_desc), "r"(smem_int_desc));
  ```
  переменная объявлена `const`, инициализирована нулём и передана как **входной** операнд
  инструкции, для которой она — приёмник. Выходного ограничения (`"=l"`) нет. Дальше её же
  берут входом ещё восемь `asm volatile` блоков (`:371-414`). Это UB: корректность держится
  на том, что компилятор материализует константу один раз в один регистр и переиспользует
  его. Это единственная запись девайс-сайда в grouped-пути, плотный GEMM её не зовёт — то
  есть отказ будет виден **только** в grouped, что в точности совпадает с симптомом #3096.
  Утверждать, что это и есть причина, без сборки нельзя; фиксирую как код-факт.

### Чего CUTLASS на sm_120 не делает (вопрос 5)

- Мультикаст кластера запрещён во всех пяти билдерах:
  `static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch")`
  (`builders/sm120_blockscaled_mma_builder.inl:104`, `sm120_mma_builder.inl:84`,
  `sm120_array_mma_builder.inl:87`, `sm120_sparse_mma_builder.inl:163`,
  `sm120_blockscaled_sparse_mma_builder.inl:179`). Совпадает с нашим замером
  «multicast в 600 раз медленнее».
- Нет wgmma: `sm120_blockscaled_mma_tma.hpp:221-223` —
  `static_assert(not is_base_of<GMMA::DescriptorIterator, FrgTypeA> && ..., "MMA atom must
  source both A and B operands from rmem")`. Замена — классический
  TMA→smem→`ldmatrix`→`mma.sync`.
- Нет tcgen05: `cute/arch/config.hpp:179-186` — `CUTE_ARCH_TCGEN05_TMEM_ENABLED` даётся
  только SM100A/101A/103A/107A. Замена аккумулятора в TMEM — регистры:
  `sm90_gemm_array_tma_warpspecialized_cooperative.hpp:170-173` считает
  `RegsPerThread = 2·TileM·TileN/NumMmaThreads·sizeof(acc)/4`, и потолок регистров math-WG
  зафиксирован в 232 (`:187`), load-WG — 40.
- **f16-аккумулятор на block-scaled отсутствует и в CUTLASS**: `grep -c "TN_VS<.*half_t"
  include/cute/arch/mma_sm120.hpp` → **0**. При этом у не-block-scaled `SM120_16x8x32_TN`
  half_t-варианты есть (`mma_sm120.hpp:911-1762`). Это независимое подтверждение брифа со
  стороны эталонной реализации.

### Разреженность 2:4 (вопрос 6)

- Инструкция есть: `cute/arch/mma_sm120_sparse.hpp:3377-3421` —
  `mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::4X.m16n8k128
  .row.col.f32.e2m1.e2m1.f32.ue4m3`, регистры `A[4] B[4] C[4] D[4] E[1] SFA[1] SFB[1]`.
- **Но**: `mma_sm120_sparse.hpp:3403` —
  `CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for NVF4 with e2m1
  and scale factor e4m3")`, и билдер это подтверждает:
  `builders/sm1xx_common.inl:171-172` — `KernelSparseTmaWarpSpecializedNvf4Sm120 → return 32`
  (плотный `KernelTmaWarpSpecializedNvf4Sm120 → return 16`, `:144-159`).
- Метаданные: `builders/sm1xx_sparse_config.inl:116-118` —
  `TensorEAtom_MMA_F4 = Layout<Shape<_128,_256>, Stride<_256,_1>>`, `ElementEMmaSparsity=16`
  → 16 байт на 256 логических элементов K = **0.5 бита на логический элемент**.
  Выравнивание: `:138-139` M кратно 128, K кратно 256; `:148-151` для F4 K-мажорного —
  `TensorAAlignmentK = 128·2 = 256`.
- Компрессор: `cutlass/include/cutlass/transform/kernel/sparse_gemm_compressor.hpp:301`
  (специализация под `arch::Sm120`) — отдельное ядро, которое из плотного тензора делает
  сжатый A + E.
- Тесты: `test/unit/gemm/device/sm120_blockscaled_sparse_tensorop_gemm/` (7 файлов),
  пример `examples/80_blackwell_geforce_sparse_gemm/80b_..._nvfp4_nvfp4_sparse_gemm.cu:86-109`.

### Асимметричная буферизация A/B (найдено попутно, самое интересное)

- `builders/sm120_blockscaled_sparse_mma_builder.inl:98-120` — если в 99 КиБ не влезают даже
  2 полные стадии, K-плитка **одного** операнда режется вдвое и ему даётся 3 полустадии:
  «instead of buffering K=256 with 2 stages, it uses K=128, with 3 stages. From the kernel's
  TileK view (K=256), B is 1.5 stages… A/B is with asymmetric DMA and buffering, as they are
  with different TileK and buffer advance steps».
- `sm120_blockscaled_sparse_mma_tma.hpp:156-181` — два независимых конвейера
  `MainloopPipelineMK<StagesA>` и `MainloopPipelineNK<StagesB>` с разными `TileShape`/`TileShapeB`
  (`:159` `AsymmetricKRatio = StagesA != StagesB ? 2 : 1`).
- `kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp:119-135` — два
  отдельных варпа-продюсера `LoadMK` и `LoadNK`.
- `sm120_blockscaled_sparse_mma_tma.hpp:1223-1295` — тело: NK-конвейер продвигается **дважды**
  за один проход MK-конвейера, между половинами — `NamedBarrier::sync`.
- И самый радикальный вариант того же: `:235, 1303-1310` — если стадий всё равно не хватает,
  метаданные E вообще **выкидываются из shared** (`UseSmemE=false`) и читаются потребителем
  прямо из глобальной памяти/L2 (`gemm_loop_with_GmemE`).

### Как этим пользуются соседи

- vLLM: `vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu:54-73` —
  всего две конфигурации, `Shape<_128,_128,_128>` и `Shape<_256,_128,_128>`, cluster 1×1×1,
  `KernelScheduleAuto`. Grouped NVFP4 на SM120 они через CUTLASS не гоняют.
- FlashInfer держит **свой** sm120-стек mxfp8/fp8 (не CUTLASS-коллективы) в
  `flashinfer/csrc/cute_sm120_mxfp8_groupwise/`. Оттуда:
  - `sm120_common/moe_tile_selection.h:29-38` — выбор плитки по волнам:
    `cost(tile_m) = ceil(tiles/num_sms) · (tile_m + 48)`.
  - `cute_sm120_mxfp8_runner.cu:49-82, 84-98` — тактика выбирается по **строкам на эксперта**
    (`total_rows/num_experts`): ≤8 → SwapAB-плитка `<128, 8, 128>` (токены уходят в N),
    ≤32 → TileM=32, дальше 64/128 по счёту волн.
  - `sm120_common/moe_scheduler.cuh:108-200` — **выделенный варп-планировщик**: один варп
    разрешает плитки вперёд и публикует их в mbarrier-кольцо `MoeSchedStorage`, потребители
    только снимают готовые `MoeWorkTile`.
  - `sm120_blockscaled/sf_mxfp8_tma_load.cuh:96-112` — третья раскладка масштабов:
    **MN-мажорная** `make_layout(make_shape(scale_m, scale_k, L), make_stride(_1, scale_m,
    scale_m·scale_k))`, где один элемент `ElementSFLoad` (int32) упаковывает
    `PACK_NSF = 4` подряд идущих K-групп одной строки.
  - `sm120_fused_moe/fp8_builder.cuh:52-77` — gate и up считаются одной плиткой активаций
    (`smem_B_up` + `smem_B_gate`), при SwapAB shared-память mainloop и эпилога объединяется
    (`kUnionSmem = SwapAB_`).

## Что у нас сегодня

- Инструкция та же: `ninfer/src/ops/common/mma.cuh:83-104` (`mma_nvfp4_e4m3`).
  `mma.cuh` целиком содержит 6 обёрток; **f16-аккумулятора для fp8 нет**
  (`mma.cuh:60-66` — только `.f32.e4m3.e4m3.f32`), разреженных MMA нет вообще
  (grep `mma.sp` / `m16n8k128` по `src/` — пусто).
- Плотный NVFP4 GEMM: `ninfer/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:84-107` —
  `BlockM ∈ {128,256}`, `BlockN = 128`, **`BlockK = 128`**, `Stages ∈ [2,4]`,
  варпы 4×2 (`kWarpsM=4, kWarpsN=2`), `kK64PerStage = 2`.
- Свизл кодов — `CU_TENSOR_MAP_SWIZZLE_64B` (`nvfp4_w4a4_tma.cuh:69, 73`), вручную
  повторён в потребителе как `((logical_byte>>4) ^ ((row>>1)&3))*16 + (logical_byte&15)`
  (`:277-279, 297-299`).
- **Масштабы активаций** — K-мажорные, `(token, group)` row-major, пишутся нашим же ядром:
  `ninfer/src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh:406`
  `scales[token * kGroupsPerRow + group] = quantized.scale`.
  TMA-дескриптор: `nvfp4_w4a4_tma.cuh:74-78`, бокс `16 × BlockM`, `SWIZZLE_NONE`,
  с комментарием `:58-60`: «TMA's innermost box is at least one 16-byte transaction. A K128
  tile consumes the first eight bytes of each row; the second half is harmless look-ahead».
  Загрузка `:230-231` идёт по координате `(k_tile/2)*16`, то есть **одни и те же 16 байт
  TMA-ятся дважды**, на чётном и на нечётном k_tile; в транзакции
  (`:220`) заявлено `BlockM·16` байт при нужных `BlockM·8`.
- Масштабы весов — плотные (`:79-81`, бокс 16×64 = 1024 подряд идущих байта), тут потерь нет.
- Внутренний цикл: `nvfp4_w4a4_tma.cuh:268-317` — на каждом `local_k64` **сначала** грузятся
  все фрагменты и масштабы, **потом** идут 16 MMA. Между стадиями стоит
  `nvfp4_mbarrier_wait` (`:265`), после которого первый `ldmatrix` находится на критическом
  пути до первой MMA. Регистровой предзагрузки следующего K-шага нет.
- Эпилог: `:319-357` — `bar.sync`, монолитная запись всей плитки в smem (union с тензорами),
  ещё `bar.sync`, затем векторные `uint4` store. Перекрытия с хвостом mainloop нет,
  TMA-store не используется.
- MoE (это модель 35B-A3B, Q4/Q5/Q6/W8 — **не** NVFP4): grouped-GEMM устроен не через
  tensormap.replace, а через материализованный на девайсе список работ.
  `ninfer/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:172-228` — одно ядро на 256
  потоков делает префикс-сумму по экспертам, пишет `expert_offsets`, `tile_bases` и плоский
  список `route_job_experts` / `route_job_columns` / `route_job_count`.
  `:292-302` — персистентная сетка с grid-stride и тремя зависимыми глобальными чтениями на
  итерацию. Ширина колонки эксперта: `:1165-1166`,
  `route_job_bn = (tokens >= 768 ? 64 : 32)`.
- Декодовый NVFP4 путь — не тензорные ядра, а GEMV/small-T
  (`ninfer/src/ops/linear/nvfp4/nvfp4_config.h:29-105`, `nvfp4_gemv.cu`, `nvfp4_small_t.cu`).

## Кандидаты для NInfer

Все числа ниже — для 27B, слой MLP gate_up: `N=34816, K=5120`, веса
34816·5120/2 = 89.13 МБ кодов + 34816·5120/16 = 11.14 МБ масштабов = 100.27 МБ.
Порог «память/счёт»: 100.27e6/1792e9 = 55.9 мкс против 2·M·34816·5120/2021.8e12 = M·0.176 мкс
→ **M ≈ 317 токенов**. На чанке 8192 задача пересчётная с запасом ×26, на декоде (M≤8) —
памятезависимая с запасом ×40. Это определяет, какие рычаги вообще имеют смысл.

---

- **1. Регистровая конвейеризация K-шага через границу стадии.**
  CUTLASS в `sm120_blockscaled_mma_tma.hpp:868-901` располагает
  `consumer_release / ++read / consumer_wait` **перед** `gemm_kblock(k_block)` последнего
  K-шага, и сразу после ожидания делает `copy_kblock` первого шага новой стадии. То есть
  ожидание mbarrier и латентность `ldmatrix` новой стадии перекрыты MMA предыдущего шага,
  фрагменты которого уже лежат в регистрах. У нас
  (`nvfp4_w4a4_tma.cuh:265-317`) после `nvfp4_mbarrier_wait` первое, что происходит, — это
  `ldmatrix`, и его латентность голая.
  **Ожидаемый эффект.** Prefill, дольше ничего. Оценка: на плитке 128×128×128 один CTA
  за один k_tile выдаёт 8 варпов × 2 шага × 16 MMA = 256 инструкций `m16n8k64`, то есть
  256·8192 = 2.10 MMAC = 4.19 MFLOP. При пиковых 2021.8/170 = 11.89 TFLOP/s на SM это
  ≈ 846 тактов при 2.4 ГГц, по 423 такта на K-шаг. Из двух шагов стадии компилятор почти
  наверняка уже перекрывает второй (объявления массивов внутри развёрнутого цикла,
  барьера между шагами нет), а первый — не может. Открытая латентность цепочки
  «ldmatrix ×10 + LDS ×10» — порядка 40-60 тактов на 423 → **потолок ≈ 5-7% на этом ядре**.
  Доля этого ядра в префилле 27B должна быть замерена отдельно (в памяти записано, что
  60.4% префилла сидит на ярусе `mma_bf16`, а не на fp4).
  **Чем меряем.** Стенд: `linear` NVFP4 gate_up, `N=34816, K=5120`, T=8192, два прогона.
  A/B: сборка с двойным набором `a_fragments/b_fragments/a_scales/b_scales` и переносом
  загрузок шага `i+1` перед MMA шага `i`, включая перенос через `mbarrier_wait`.
  Приёмка — SASS: 32 `HMMA`-подобные инструкции в теле без вклинившихся `LDSM`, плюс
  побитово равный выход.
  **Риски.** +34 регистра на поток (сейчас: 64 аккумулятора + 34 фрагмента ≈ 138 из 232 у
  math-варпов после `setmaxnreg.inc 232`, `nvfp4_w4a4_tma.cuh:243`). Должно влезть, но при
  `BlockM=256` (`kMmaM=4`) прибавка 4·4+... больше — проверять отдельно. Численность не
  меняется, выход обязан остаться побитовым.
  **Объём.** 1-2 файла: `nvfp4_w4a4_tma.cuh` и симметрично
  `linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh`.

- **2. `BlockK = 256` вместо 128: 128-байтовый свизл и вдвое меньше барьеров.**
  `builders/sm120_common.inl:125-135` выбирает `Layout_K_SW128_Atom` (128 Б/строку,
  `Swizzle<3,4,3>`), как только `TileK` делится на 256 элементов fp4;
  при нашем `TileK=128` доступен только `SW64`. NVIDIA поставляет обе ширины
  (`generator.py:11394-11416`: K∈{128,256}), а канон юнит-тестов — 256.
  Механизм: `kK64PerStage` растёт с 2 до 4, `Stages` падает с 4 до 2 — **та же глубина K в
  полёте (512) и та же shared-память**, но вдвое меньше `mbarrier.arrive/try_wait`,
  строка TMA 128 Б вместо 64 Б (совпадает с сектором L2), и вчетверо длиннее участок MMA
  между барьерами, что усиливает кандидат 1.
  **Ожидаемый эффект.** Prefill. Барьеров на K=5120: 40 против 20. При ~30 тактах на пару
  arrive/wait это 1200 из ≈ 33.8 тыс. тактов MMA → **≈1.8%** только от барьеров; выигрыш от
  128-байтовых строк TMA/`ldmatrix` не оценивается аналитически, меряется.
  **Чем меряем.** Тот же стенд, вариант `Nvfp4W4a4TmaSchedule<128, 256, 2, ...>` с
  `kBlockK=256`, `kK64PerStage=4`, `kCodeRowBytes=128`, `SWIZZLE_128B` в четырёх
  дескрипторах и пересчитанной формулой свизла в потребителе.
  **Риски.** Формула свизла в `:277-279/:297-299` жёстко закодирована под 64B — её нужно
  переписать под `Swizzle<3,4,3>` (XOR 3 бита), это самая вероятная точка ошибки; выход
  обязан остаться побитовым. Shared на стадию 36.9 КиБ → при `BlockM=256` две стадии
  не влезут, вариант только для `BlockM=128`.
  **Объём.** 2 файла (те же).

- **3. MN-мажорная упаковка масштабов активаций.**
  Мы платим ×2 по SFA-трафику и по SFA-месту в shared только потому, что наш SF-план
  K-мажорный, а минимальный внутренний бокс TMA — 16 байт
  (`nvfp4_w4a4_tma.cuh:58-60, 74-78, 220, 230-231`). FlashInfer решает это транспонированием:
  `sf_mxfp8_tma_load.cuh:96-112` — план `(scale_m, scale_k)` со страйдом `(1, scale_m)`,
  где один 32-битный элемент держит 4 подряд идущих K-группы одной строки. Тогда бокс
  «BlockM строк × 1 слово» — это `BlockM·4` **подряд идущих** байт, плотно при любой ширине
  K-плитки. Если внутреннего измерения не хватит (лимит 256 элементов,
  `cute/atom/copy_traits_sm90_tma.hpp:797`) — кодировать дескриптор более широким типом,
  как CUTLASS делает через `make_tma_copy<uint16_t>` (`sm120_blockscaled_mma_tma.hpp:309`).
  Побочно снимается конфликт банков: сейчас 32 лейна читают 16 различных адресов
  `4·scale_row + j` по 8 банкам (2-way конфликт, занято 8 из 32 банков); при MN-мажорном
  плане адрес `k·BlockM + scale_row` даёт 16 банков по одному адресу — бесконфликтно.
  **Ожидаемый эффект. Честно: малый.** За k_tile CTA заливает 19456 Б в shared за ≈846
  тактов = 23 Б/такт — это далеко не узкое место; лишние 1024 Б (5.3%) в пересчётном режиме
  не видны. Экономия банков — 2 такта на 423. Реальная ценность — освобождение 1024 Б/стадию
  shared (4 КиБ при 4 стадиях) и снятие ограничения, мешающего кандидату 2. **Брать только
  как часть 2, отдельно не окупается.**
  **Чем меряем.** A/B того же ядра при неизменном всём остальном; ожидание — 0..1%.
  Формат на диске не меняется: масштабы активаций пишет наше же ядро
  (`nvfp4_w4a4_mma.cuh:406`), веса не трогаются.
  **Риски.** Меняется только адрес записи в квантователе и адрес чтения в потребителе;
  байты масштабов те же → выход побитовый. Три места правки.
  **Объём.** 3 файла.

- **4. Не тащить к себе девайс-сайдный `tensormap.replace` для MoE (защитный вывод).**
  Наш grouped-GEMM построен на материализованном списке работ
  (`sparse_moe_prefill_kernels.cu:172-228, 292-302`) и не переписывает TMA-дескрипторы
  вообще. Это ровно тот класс, в котором живёт #3096: подмена включается арх-условным
  макросом (`arch/config.h:62-69`), при его отсутствии деградирует в тихий no-op
  (`cute/config.hpp:160`), а сама реализация подмены размеров содержит UB-конструкцию с
  входным операндом вместо выходного (`cute/arch/copy_sm90_desc.hpp:366-370`).
  **Ожидаемый эффект.** Нулевой по скорости, положительный по риску: закрывает направление
  «сделать MoE как ptr-array grouped GEMM с TMA-дескрипторами на эксперта».
  **Чем меряем.** Не меряется; это отказ от направления.
  **Объём.** 0 файлов.

- **5. Выделенный варп-планировщик перед персистентным циклом MoE (низкий приоритет).**
  `flashinfer/.../sm120_common/moe_scheduler.cuh:135-200` — один варп разрешает плитки
  вперёд и публикует их в mbarrier-кольцо, потребители снимают готовое. У нас каждая
  итерация grid-stride цикла начинается с трёх зависимых глобальных чтений
  (`sparse_moe_prefill_kernels.cu:297-301`: `route_job_experts[job]`,
  `expert_offsets[expert]`, `expert_offsets[expert+1]`, `route_job_columns[job]`).
  **Ожидаемый эффект.** Prefill MoE (35B). Это ~2 зависимых промаха L2 (≈2×250 тактов) на
  плитку; при плитке в тысячи тактов — доли процента. Смысл появляется только если плитка
  короткая (узкие эксперты, d_ff 512).
  **Чем меряем.** Стенд MoE prefill 35B, T=768..8192; A/B — предвыборка полей следующей
  работы в регистры в конце текущей итерации (дешёвый вариант того же), без отдельного варпа.
  **Риски.** Практически нет; чистая перестановка чтений.
  **Объём.** 1 файл.

- **6. Асимметричная K-плитка A/B — направление на будущее, сегодня неприменимо.**
  Механизм (`builders/sm120_blockscaled_sparse_mma_builder.inl:98-120`,
  `sm120_blockscaled_sparse_mma_tma.hpp:156-181, 1223-1295`) даёт «полторы стадии» одному
  операнду, когда 99 КиБ не хватает на две полные. В нашем плотном NVFP4 GEMM A и B
  симметричны (обе 4-битные, BlockM=BlockN=128), выигрыша нет. Он появился бы в
  `linear_swiglu`, где B-сторона удвоена (`nvfp4_linear_swiglu_w4a4_tma.cuh:26-27` —
  `b_scales[stage][2][...]`, две плитки весов на одну плитку активаций): там A мог бы жить
  в 2 стадиях по K=256, а удвоенное B — в 4 стадиях по K=128. **Записываю как открытое
  направление, а не как заявку**: без замера доли этого ядра оценивать нечего, а брифом
  «глубокие async-конвейеры» уже отвергнуты (−9..15%), и это близкий родственник.

## Опровергнуто / не переносится

- **Раскладка масштабов CUTLASS (блок 128×4) не экономит нам чтений.**
  Отображение «лейн → строка масштаба» задано ISA и у нас уже точно такое же
  (`mma_traits_sm120.hpp:158-162` против `nvfp4_w4a4_tma.cuh:257-258`). Единственная
  свобода — адрес 4-байтового слова в shared, и там блок 128×4 даёт **тот же** паттерн
  банков, что у нас: адрес `16·(m mod 32) + 4·(m div 32)`, варп читает 16 строк одной
  32-строчной группы → 8 банков по 2 адреса, 2-way конфликт. Смысл их формата — плотный
  TMA-бокс и однобайтовый `UniversalCopy` вместо `ldmatrix`, а не банки. Для нас лучше
  MN-мажорный вариант FlashInfer (кандидат 3), а не блочный CUTLASS.
- **Тензорные ядра на декоде (M=1..8) через SwapAB-плитку `<128,8,K>`.** CUTLASS и
  FlashInfer оба ставят токены в N при малом M (`generator.py:11395-11396`,
  `cute_sm120_mxfp8_runner.cu:71-73`). Нам это не даёт ничего: при M=8 задача читает 100.27 МБ
  весов = 55.9 мкс при пике HBM 1792 ГБ/с, а MMA той же задачи на ярусе 2021.8 TOPS —
  1.41 мкс. Запас памятезависимости ×40; наш GEMV
  (`nvfp4_config.h:29-38`, `nvfp4_gemv.cu`) выбран правильно, ярус MMA на декоде
  нерелевантен по построению.
- **Групповой tile-scheduler CUTLASS (`sm90_tile_scheduler_group.hpp:314-395`).** Красивый
  лейн-параллельный скан по 32 группам, но он решает задачу, которой у нас нет: он ищет
  владельца плитки на лету, потому что список работ не материализован. У нас список
  строится один раз ядром на 256 потоков (`sparse_moe_prefill_kernels.cu:172-228`) — это
  строго дешевле.
- **Выбор ширины колонки эксперта по числу строк на эксперта.** Хотел предложить заменить
  наше `tokens >= 768 → route_job_bn=64` (`sparse_moe_prefill_kernels.cu:1165-1166`) на
  флешинферовскую метрику `total_rows/num_experts`
  (`cute_sm120_mxfp8_runner.cu:84-98`). **Не переносится**: при фиксированной модели
  (256 экспертов, top-8) `assignments/experts = tokens/32`, то есть наш порог по токенам —
  это тот же порог по строкам на эксперта с точностью до константы (768 токенов ↔ 24 строки
  на эксперта). Единственное, чего наша формула не видит, — счётчик SM в терме волн
  (`moe_tile_selection.h:34-37`), а при 256 экспертах и 170 SM волн всегда ≥2, так что и он
  почти ничего не меняет. Заявку снимаю.
- **`fp4_shift` + `ldmatrix .b8x16.b4x16_p64`** (`mma_traits_sm120.hpp:210-231`,
  `builders/sm120_common.inl:55-68` → `SM100_SU4_DU8x16_x4_LDSM_N`). Это костыль для случая,
  когда fp4 кормят инструкции `kind::f8f6f4` с 8-битными контейнерами. Мы используем родную
  `mxf4nvf4` m16n8k64, где сдвиг не нужен — CUTLASS на этом пути его тоже не ставит.
  Переносить нечего.
- **2:4 nvfp4 (ярус 3986.2 TOPS) — честная оценка: сегодня закрыт, и не по кернельным
  причинам.** Арифметика привлекательная: коды падают вдвое (89.13 → 44.57 МБ), метаданные
  добавляют 34816·5120·0.5/8 = 11.14 МБ, масштабы при VS=32 падают вдвое (11.14 → 5.57 МБ);
  итого 61.28 МБ против 100.27 МБ = **0.61×** по трафику весов, то есть декодовый пол
  55.9 → 34.2 мкс, при удвоенном пике MMA. Но цена входа:
  (а) `mma_sm120_sparse.hpp:3403` и `builders/sm1xx_common.inl:171-172` требуют
  **SFVecSize = 32**, а наш чекпоинт квантован группой 16 → полный реквант с более грубым
  масштабом и новый прогон точностной батареи;
  (б) нужен 2:4-прунинг весов с восстановлением качества — это не кернельная работа;
  (в) `builders/sm1xx_sparse_config.inl:116-151` требует компрессор (есть готовый:
  `transform/kernel/sparse_gemm_compressor.hpp:301`) и выравнивание M кратно 128, K кратно 256
  (наши 34816 и 5120 проходят);
  (г) новое семейство ядер с метаданными E, включая обработку случая «E не влезает в shared».
  Рекомендация: **не начинать**, пока не пройдёт отдельный офлайн-эксперимент по точности
  «2:4 + NVFP4 группой 32» на нашей батарее. Кернельная часть — самая дешёвая часть этой темы.
- **Мультикаст, wgmma, tcgen05, f16-аккумулятор на block-scaled** — CUTLASS их на sm120 не
  использует ровно по тем же причинам, что записаны в брифе, и подтверждает это
  статик-ассертами и отсутствием специализаций (см. раздел «вопрос 5»). Новой информации нет,
  но бриф теперь подтверждён независимо.

## Открытые вопросы

1. **Доля NVFP4-ядер в префилле 27B.** Все кандидаты 1-3 бьют в одно ядро
   (`nvfp4_w4a4_tma_kernel` и его SwiGLU-близнеца). В памяти записано, что 60.4% префилла
   сидит на ярусе `mma_bf16` — если это верно и для 27B, потолок всех трёх кандидатов
   вместе — единицы процентов от TTFT. Мерить долю надо до, а не после.
2. **Перекрывает ли компилятор загрузки второго K-шага уже сейчас.** Оценка эффекта
   кандидата 1 (5-7%) целиком держится на предположении, что перекрывается только один шаг
   из двух. Проверяется чтением SASS, чего в этой ячейке сделать нельзя (сборка запрещена).
3. **Реальная причина #3096.** Автор issue называет `compute_120a` vs `compute_120f`; код
   говорит обратное (`arch/config.h:62-69`, `cute/arch/config.hpp:131-137`). Разрешить это
   можно только сборкой обоих вариантов, чего мы не делаем. Ответа NVIDIA в треде нет.
   Практический вывод от этого не зависит: направление всё равно отвергнуто (кандидат 4).
4. **Стоит ли `BlockN < 128` для NVFP4.** CUTLASS поставляет N=8/16/32/64
   (`generator.py:11394-11406`) и меняет под них и `AtomLayoutMNK`
   (`builders/sm120_blockscaled_mma_builder.inl:131-134`), и copy-atom B
   (`builders/sm120_common.inl:80-104`). У нас `kBlockN` жёстко 128 и в `linear`, и в
   `linear_swiglu` (`nvfp4_w4a4_tma.cuh:91`, `nvfp4_linear_swiglu_w4a4_tma.cuh:17`).
   На наших N (34816, 17408, 14336, 5120) узкая N-плитка нужна не для формы задачи, а как
   способ поднять число CTA при малом M — но малый M у нас уходит на GEMV. Не разобрано.
5. **Эпилог.** CUTLASS дробит эпилог на подплитки 64×32 и перекрывает TMA-store с хвостом
   mainloop (`epilogue/collective/builders/sm120_builder.inl:94-104, 191-197`); мы делаем
   монолитную запись между двумя `bar.sync` (`nvfp4_w4a4_tma.cuh:319-357`). При K=5120
   эпилог — малая доля, при коротком K (MoE, d_ff 512) — уже нет. Не измерено.
