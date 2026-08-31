#!/usr/bin/env python3
"""Г6: проекция гейтинга GDN идёт параллельно с qkvz на втором потоке.

В слое GDN порядок такой: норма -> gating_proj_partial (240 CTA, 3.15 мкс) ->
gating_proj_reduce (2 CTA, 1.71 мкс) -> qkvz (1024 CTA, 56 мкс). Вторая и третья ступени зависят
только от нормы, ровно как и qkvz, но занимают машину почти вхолостую: 233 мкс за раунд = 1.42%
при том, что читают меньше мегабайта весов.

Правка: норма считается на основном потоке, гейтинг уходит на боковой поток через событие, а
соединение ставится перед первым потребителем g/beta — рекуррентом. Между форком и соединением
на основном потоке успевают qkvz, свёртка и запись состояния.

Рабочая область гейтинга не может браться из общей арены: основной поток в это же время
раздаёт из неё память под qkvz. Поэтому в DeviceContext заводится небольшой отдельный буфер.

Выключатель: NINFER_GDN_GATING_FORK=0."""
import sys

R = '/root/ninfer_d4'


def patch(path, pairs):
    p = f'{R}/{path}'
    s = open(p, encoding='utf-8').read()
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f'{path}: якорь встречается {s.count(old)} раз:\n---\n{old[:180]}\n---')
        s = s.replace(old, new, 1)
    open(p, 'w', encoding='utf-8').write(s)
    print('пропатчен', path)


# --- 1. DeviceContext: боковой поток, два события и буфер под рабочую область гейтинга -------
patch('src/core/device.h', [
    ("""    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};""",
     """    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Боковой поток для веток, независимых от основной цепочки слоя (см. Г6: проекция гейтинга
    // GDN зависит только от нормы и идёт параллельно с qkvz). События — форк и соединение.
    cudaStream_t side_stream          = nullptr;
    cudaEvent_t side_fork             = nullptr;
    cudaEvent_t side_join             = nullptr;
    void* side_scratch                = nullptr;
    std::size_t side_scratch_bytes    = 0;
    cudaDeviceProp props{};"""),
])

patch('src/core/device.cu', [
    ("""    stream          = compute;
    transfer_stream = load;
}""",
     """    cudaStream_t side = nullptr;
    err               = cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(side_stream) failed", err));
    }

    cudaEvent_t fork_event = nullptr;
    cudaEvent_t join_event = nullptr;
    void* scratch          = nullptr;
    // Рабочая область ветки на боковом потоке. Гейтингу GDN хватает десятков килобайт;
    // берём с запасом, чтобы не пересчитывать при смене расписания.
    constexpr std::size_t kSideScratchBytes = 4u << 20;
    if (cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc(&scratch, kSideScratchBytes) != cudaSuccess) {
        if (fork_event != nullptr) { (void)cudaEventDestroy(fork_event); }
        if (join_event != nullptr) { (void)cudaEventDestroy(join_event); }
        if (scratch != nullptr) { (void)cudaFree(scratch); }
        destroy_stream(side);
        destroy_stream(load);
        destroy_stream(compute);
        throw std::runtime_error("failed to create the side-branch stream resources");
    }

    stream             = compute;
    transfer_stream    = load;
    side_stream        = side;
    side_fork          = fork_event;
    side_join          = join_event;
    side_scratch       = scratch;
    side_scratch_bytes = kSideScratchBytes;
}"""),
    ("""DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}""",
     """DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    if (side_scratch != nullptr) { (void)cudaFree(side_scratch); }
    if (side_join != nullptr) { (void)cudaEventDestroy(side_join); }
    if (side_fork != nullptr) { (void)cudaEventDestroy(side_fork); }
    destroy_stream(side_stream);
    destroy_stream(transfer_stream);
    destroy_stream(stream);
    side_scratch       = nullptr;
    side_scratch_bytes = 0;
    side_join          = nullptr;
    side_fork          = nullptr;
}"""),
    ("""DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}""",
     """DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      side_stream(other.side_stream), side_fork(other.side_fork), side_join(other.side_join),
      side_scratch(other.side_scratch), side_scratch_bytes(other.side_scratch_bytes),
      props(other.props) {
    other.stream             = nullptr;
    other.transfer_stream    = nullptr;
    other.side_stream        = nullptr;
    other.side_fork          = nullptr;
    other.side_join          = nullptr;
    other.side_scratch       = nullptr;
    other.side_scratch_bytes = 0;
}"""),
])
print('device: боковой поток заведён')

# --- 1b. Присваивание переносом ---------------------------------------------------------------
patch('src/core/device.cu', [
    ("""    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;""",
     """    if (side_scratch != nullptr) { (void)cudaFree(side_scratch); }
    if (side_join != nullptr) { (void)cudaEventDestroy(side_join); }
    if (side_fork != nullptr) { (void)cudaEventDestroy(side_fork); }
    destroy_stream(side_stream);

    device             = other.device;
    props              = other.props;
    stream             = other.stream;
    transfer_stream    = other.transfer_stream;
    side_stream        = other.side_stream;
    side_fork          = other.side_fork;
    side_join          = other.side_join;
    side_scratch       = other.side_scratch;
    side_scratch_bytes = other.side_scratch_bytes;

    other.stream             = nullptr;
    other.transfer_stream    = nullptr;
    other.side_stream        = nullptr;
    other.side_fork          = nullptr;
    other.side_join          = nullptr;
    other.side_scratch       = nullptr;
    other.side_scratch_bytes = 0;
    return *this;"""),
])

# --- 2. Хук варианта: проекция гейтинга от уже нормированного входа ----------------------------
HOOK_DECL = """    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, cudaStream_t stream);"""
HOOK_DECL_NEW = HOOK_DECL + """

    // Та же проекция, но от уже нормированного входа: нужна, чтобы норму и гейтинг можно было
    // развести по разным потокам (Г6).
    static void gdn_control_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                       Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                       cudaStream_t stream);"""


def hook_body(variant_ns):
    if variant_ns == "qwen3_6_35b_a3b":
        return """
void Variant::gdn_control_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    ops::gdn_gating_proj(hidden, weights.a_b_projection, weights.a_log, weights.dt_bias, workspace,
                         g, beta, stream);
}
"""
    return """
void Variant::gdn_control_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_gating_proj(hidden, split->a_projection, split->b_projection, weights.a_log,
                             weights.dt_bias, workspace, g, beta, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_gating_proj(hidden, fused, weights.a_log, weights.dt_bias, workspace, g, beta, stream);
}
"""


for tgt in ('qwen3_6_27b', 'qwen3_6_35b_a3b'):
    s = open(f'{R}/src/targets/{tgt}/impl/variant.h', encoding='utf-8').read()
    if s.count(HOOK_DECL) != 1:
        sys.exit(f'{tgt}/variant.h: объявление хука встречается {s.count(HOOK_DECL)} раз')
    open(f'{R}/src/targets/{tgt}/impl/variant.h', 'w', encoding='utf-8').write(
        s.replace(HOOK_DECL, HOOK_DECL_NEW, 1))
    p = f'{R}/src/targets/{tgt}/impl/variant.cpp'
    s = open(p, encoding='utf-8').read()
    anchor = 'void Variant::post_mixer('
    if s.count(anchor) != 1:
        sys.exit(f'{tgt}/variant.cpp: якорь post_mixer встречается {s.count(anchor)} раз')
    open(p, 'w', encoding='utf-8').write(s.replace(anchor, hook_body(tgt).strip() + '\n\n' + anchor, 1))
    print('пропатчен вариант', tgt)

# --- 3. Форк и соединение в gdn_mix ------------------------------------------------------------
patch('src/targets/qwen3_6/impl/runtime/text_context_impl.h', [
    ("""bool mtp_sampled_draft_enabled() {""",
     """bool gdn_gating_fork_enabled() {
    static const bool enabled = [] {
        // Включено по умолчанию: проекция гейтинга не зависит ни от чего, кроме нормы, и на
        // основном потоке стоит 1.42% раунда почти вхолостую. Выключатель на случай отладки.
        const char* value = std::getenv("NINFER_GDN_GATING_FORK");
        return value == nullptr || value[0] != 0x30;
    }();
    return enabled;
}

bool mtp_sampled_draft_enabled() {"""),
    ("""    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);""",
     """    // Г6: гейтинг зависит только от нормы, поэтому уходит на боковой поток и идёт параллельно
    // с qkvz, свёрткой и записью состояния. Соединение — перед первым потребителем g/beta.
    const bool gating_forked = gdn_gating_fork_enabled() && ctx_.side_stream != nullptr &&
                               ctx_.side_scratch != nullptr;
    if (gating_forked) {
        ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, /*unit_offset=*/true, h, s);
        CUDA_CHECK(cudaEventRecord(ctx_.side_fork, s));
        CUDA_CHECK(cudaStreamWaitEvent(ctx_.side_stream, ctx_.side_fork, 0));
        WorkspaceArena side_workspace(DeviceSpan{ctx_.side_scratch, ctx_.side_scratch_bytes});
        Variant::gdn_control_projection(h, *w.projection, g, beta, side_workspace,
                                        ctx_.side_stream);
    } else {
        Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g,
                                             beta, work_, s);
    }"""),
    ("""    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});""",
     """    if (gating_forked) {
        CUDA_CHECK(cudaEventRecord(ctx_.side_join, ctx_.side_stream));
        CUDA_CHECK(cudaStreamWaitEvent(s, ctx_.side_join, 0));
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});"""),
])
print('Г6 наложена')

# Заголовки, которые могли не быть нужны раньше.
p = f'{R}/src/targets/qwen3_6/impl/runtime/text_context_impl.h'
s = open(p, encoding='utf-8').read()
if '#include "ninfer/ops/rmsnorm.h"' not in s:
    s = s.replace('#include "ninfer/ops/gdn_gating.h"',
                  '#include "ninfer/ops/gdn_gating.h"\n#include "ninfer/ops/rmsnorm.h"', 1)
if '#include "core/device.h"' not in s:
    s = s.replace('#include "ninfer/ops/gdn_gating.h"',
                  '#include "core/device.h"\n#include "ninfer/ops/gdn_gating.h"', 1)
open(p, 'w', encoding='utf-8').write(s)
print('заголовки проверены')
