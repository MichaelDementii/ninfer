#!/usr/bin/env python3
"""П7: свёртка GDN в префилле пишет q, k и v сама.

Сейчас свёртка пишет слитый буфер, а потом три `extract_bf16_columns` режут его на q, k и v.
Каждый из них — `cudaMemcpy2DAsync` device-to-device с шагом 20480 Б и строкой 4096/12288 Б.
В трассе это 2304 копии на прогон: по 16.78 МБ на q и k и 50.33 МБ на v для каждого из 48 слоёв
на каждом из восьми чанков — 32 ГБ за префилл, 72.7 мс = 0.86% времени.

Плюс сам слитый приёмник: рецепт выделяет два тензора `convolution_dim x tokens`, и второй нужен
только чтобы из него резать — 83.9 МБ рабочей области на чанке 4096.

Ядро уже пишет по паре каналов на нить (`out2[t * C2 + p]`), а границы сплита — `key_dim` и
`2 * key_dim` — кратны 256, то есть попадают на границу блока: ветвление внутри блока однородно.
Достаточно дать ядру три указателя и выбрать назначение по номеру канала. Арифметика на значениях
не меняется вовсе: те же четыре `causal_conv1d_acc_pair`, тот же `silu`, та же пара bf16 —
меняется только адрес записи, поэтому результат обязан быть побитово прежним.

Это тот же механизм, что в уже смердженном PR #99 («GDN conv writes q/k/v»), применённый к
префилльному маршруту, где он не был применён: там форма snapshot (B=1..8, W=1..16) пишет
query/key/value напрямую, здесь плотный [C,T] маршрут по-прежнему режет копиями.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch, R  # noqa: E402


def insert_before(path, anchor, text):
    full = '%s/%s' % (R, path)
    s = open(full, encoding='utf-8').read()
    if s.count(anchor) != 1:
        sys.exit('%s: якорь встречается %d раз' % (path, s.count(anchor)))
    s = s.replace(anchor, text.strip() + '\n\n' + anchor, 1)
    open(full, 'w', encoding='utf-8').write(s)
    print('дополнен ' + path)


# --- 1. ядра -----------------------------------------------------------------------------------
KERNELS = r'''
// Three-destination prefill forms. Value arithmetic is identical to the fused kernels above; only
// the store address differs, so the produced numbers are bit-identical. The caller passes the row
// count of the first two destinations and the thread picks its destination from its own channel.
__global__ void
causal_conv1d_prefill_split_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                   const __nv_bfloat16* conv_state, __nv_bfloat16* out_q,
                                   __nv_bfloat16* out_k, __nv_bfloat16* out_v, std::int32_t C,
                                   std::int32_t query_rows, std::int32_t key_rows, std::int32_t T) {
    const std::int64_t C64      = static_cast<std::int64_t>(C);
    const std::int64_t c_blocks = div_up(C64, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block    = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t        = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t c_base   = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t c64      = c_base + threadIdx.x;
    if (t >= T || c64 >= C64) { return; }

    const std::int32_t c       = static_cast<std::int32_t>(c64);
    const std::int64_t src_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x0     = (t >= 3) ? x[static_cast<std::int64_t>(t - 3) * C64 + c64]
                                          : conv_state[static_cast<std::int64_t>(t) * C64 + c64];
    const __nv_bfloat16 x1     = (t >= 2) ? x[static_cast<std::int64_t>(t - 2) * C64 + c64]
                                          : conv_state[static_cast<std::int64_t>(t + 1) * C64 + c64];
    const __nv_bfloat16 x2     = (t >= 1) ? x[static_cast<std::int64_t>(t - 1) * C64 + c64]
                                          : conv_state[static_cast<std::int64_t>(t + 2) * C64 + c64];
    const __nv_bfloat16 x3     = x[src_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weight[c]) * __bfloat162float(x0);
    acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(x1);
    acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(x2);
    acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x3);
    const __nv_bfloat16 value = __float2bfloat16_rn(silu(acc));

    const std::int64_t q64 = static_cast<std::int64_t>(query_rows);
    const std::int64_t k64 = static_cast<std::int64_t>(key_rows);
    if (c64 < q64) {
        out_q[static_cast<std::int64_t>(t) * q64 + c64] = value;
    } else if (c64 < q64 + k64) {
        out_k[static_cast<std::int64_t>(t) * k64 + (c64 - q64)] = value;
    } else {
        const std::int64_t value_rows                                        = C64 - q64 - k64;
        out_v[static_cast<std::int64_t>(t) * value_rows + (c64 - q64 - k64)] = value;
    }
}

// Pair form of the above. Every destination boundary is an even channel index, so a pair is never
// divided between two destinations.
__global__ void causal_conv1d_prefill_pairs_split_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* conv_state,
    __nv_bfloat16* out_q, __nv_bfloat16* out_k, __nv_bfloat16* out_v, std::int32_t C,
    std::int32_t query_pairs, std::int32_t key_pairs, std::int32_t T) {
    const std::int64_t C2        = static_cast<std::int64_t>(C / 2);
    const std::int64_t c_blocks  = div_up(C2, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block     = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t         = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t pair_base = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t p         = pair_base + threadIdx.x;
    if (t >= T || p >= C2) { return; }

    const auto* x2      = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const auto* state2  = reinterpret_cast<const __nv_bfloat162*>(conv_state);

    const std::int64_t src_idx = static_cast<std::int64_t>(t) * C2 + p;
    const __nv_bfloat162 x0    = (t >= 3) ? x2[static_cast<std::int64_t>(t - 3) * C2 + p]
                                          : state2[static_cast<std::int64_t>(t) * C2 + p];
    const __nv_bfloat162 x1    = (t >= 2) ? x2[static_cast<std::int64_t>(t - 2) * C2 + p]
                                          : state2[static_cast<std::int64_t>(t + 1) * C2 + p];
    const __nv_bfloat162 x2v   = (t >= 1) ? x2[static_cast<std::int64_t>(t - 1) * C2 + p]
                                          : state2[static_cast<std::int64_t>(t + 2) * C2 + p];
    const __nv_bfloat162 x3    = x2[src_idx];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    causal_conv1d_acc_pair(weight2[p], x0, acc0, acc1);
    causal_conv1d_acc_pair(weight2[C2 + p], x1, acc0, acc1);
    causal_conv1d_acc_pair(weight2[2 * C2 + p], x2v, acc0, acc1);
    causal_conv1d_acc_pair(weight2[3 * C2 + p], x3, acc0, acc1);
    const __nv_bfloat162 value = __floats2bfloat162_rn(silu(acc0), silu(acc1));

    const std::int64_t qp = static_cast<std::int64_t>(query_pairs);
    const std::int64_t kp = static_cast<std::int64_t>(key_pairs);
    if (p < qp) {
        reinterpret_cast<__nv_bfloat162*>(out_q)[static_cast<std::int64_t>(t) * qp + p] = value;
    } else if (p < qp + kp) {
        reinterpret_cast<__nv_bfloat162*>(out_k)[static_cast<std::int64_t>(t) * kp + (p - qp)] =
            value;
    } else {
        const std::int64_t value_pairs = C2 - qp - kp;
        reinterpret_cast<__nv_bfloat162*>(
            out_v)[static_cast<std::int64_t>(t) * value_pairs + (p - qp - kp)] = value;
    }
}
'''

insert_before('src/ops/kernel/causal_conv1d.cuh',
              '// Writes the trailing width-3 conv window after consuming the T input columns.',
              KERNELS)

# --- 2. пускатель ------------------------------------------------------------------------------
patch('src/ops/launcher/causal_conv1d.h', [
    ("""void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,""",
     """// Same extent as causal_conv1d_prefill_launch, but the convolution stores into the three
// consumer tensors directly instead of a fused destination the caller then slices.
void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out_query, Tensor& out_key, Tensor& out_value,
                                        cudaStream_t stream);
void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,"""),
])

patch('src/ops/launcher/causal_conv1d.cu', [
    ("""void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,""",
     """void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out_query, Tensor& out_key, Tensor& out_value,
                                        cudaStream_t stream) {
    constexpr int kOutputBlock    = 256;
    constexpr int kChannelBlock   = 256;
    constexpr int kPairBlock      = 256;
    const std::int32_t C          = x.ne[0];
    const std::int32_t T          = x.ne[1];
    const std::int32_t query_rows = out_query.ne[0];
    const std::int32_t key_rows   = out_key.ne[0];
    const auto x_addr             = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr             = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto in_addr            = reinterpret_cast<std::uintptr_t>(conv_state_in.data);
    const auto out_state_addr     = reinterpret_cast<std::uintptr_t>(conv_state_out.data);
    const auto query_addr         = reinterpret_cast<std::uintptr_t>(out_query.data);
    const auto key_addr           = reinterpret_cast<std::uintptr_t>(out_key.data);
    const auto value_addr         = reinterpret_cast<std::uintptr_t>(out_value.data);

    const bool pairs =
        ((x_addr | w_addr | in_addr | out_state_addr | query_addr | key_addr | value_addr) &
         (alignof(__nv_bfloat162) - 1)) == 0 &&
        (C & 1) == 0 && (query_rows & 1) == 0 && (key_rows & 1) == 0;
    if (pairs) {
        causal_conv1d_prefill_pairs_split_kernel<<<prefill_output_grid_for(C / 2, T, kPairBlock),
                                                   kPairBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data),
            static_cast<__nv_bfloat16*>(out_query.data), static_cast<__nv_bfloat16*>(out_key.data),
            static_cast<__nv_bfloat16*>(out_value.data), C, query_rows / 2, key_rows / 2, T);
        CUDA_CHECK(cudaGetLastError());
    } else {
        causal_conv1d_prefill_split_kernel<<<prefill_output_grid_for(C, T, kOutputBlock),
                                             kOutputBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data),
            static_cast<__nv_bfloat16*>(out_query.data), static_cast<__nv_bfloat16*>(out_key.data),
            static_cast<__nv_bfloat16*>(out_value.data), C, query_rows, key_rows, T);
        CUDA_CHECK(cudaGetLastError());
    }

    causal_conv1d_prefill_state_kernel<<<grid_for(C, kChannelBlock, "prefill state"), kChannelBlock,
                                         0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,"""),
])

# --- 3. публичная операция ---------------------------------------------------------------------
patch('include/ninfer/ops/causal_conv1d_silu.h', [
    ("""/**
 * Snapshot form for B independent sequences.""",
     """/**
 * Split-destination form of the distinct-state overload. Identical arithmetic and identical state
 * effect; the single [C,T] destination is replaced by three contiguous BF16 destinations
 * [Cq,T], [Ck,T] and [Cv,T] with Cq+Ck+Cv=C, receiving channel ranges [0,Cq), [Cq,Cq+Ck) and
 * [Cq+Ck,C) respectively. The three destinations are mutually disjoint and disjoint from every
 * input. Every T>=1 is accepted. Output values are bit-identical to writing the fused destination
 * and slicing it, because only the store address differs.
 */
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out_query, Tensor& out_key,
                        Tensor& out_value, cudaStream_t stream);

/**
 * Snapshot form for B independent sequences."""),
])

# --- 4. обёртка --------------------------------------------------------------------------------
patch('src/ops/wrapper/causal_conv1d_silu.cpp', [
    ("""void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream) {""",
     """void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out_query, Tensor& out_key,
                        Tensor& out_value, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 ||
        conv_state_in.dtype != DType::BF16 || conv_state_out.dtype != DType::BF16 ||
        out_query.dtype != DType::BF16 || out_key.dtype != DType::BF16 ||
        out_value.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_state/out must be BF16");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(weight, "weight");
    (void)numel_allow_zero(conv_state_in, "conv_state_in");
    (void)numel_allow_zero(conv_state_out, "conv_state_out");
    (void)numel_allow_zero(out_query, "out_query");
    (void)numel_allow_zero(out_key, "out_key");
    (void)numel_allow_zero(out_value, "out_value");

    require_x_shape(x);
    require_weight_shape(weight, x.ne[0]);
    require_state_shape(conv_state_in, x.ne[0]);
    require_state_shape(conv_state_out, x.ne[0]);
    require_split_out_shape(x, out_query, out_key, out_value);
    if (n == 0) { return; }

    require_non_empty_accessible(x, weight, conv_state_in, out_query);
    if (!out_key.is_contiguous() || !out_value.is_contiguous() || out_key.data == nullptr ||
        out_value.data == nullptr) {
        throw std::invalid_argument("causal_conv1d: split destinations must be contiguous "
                                    "and non-null");
    }
    if (!conv_state_out.is_contiguous() || conv_state_out.data == nullptr) {
        throw std::invalid_argument(
            "causal_conv1d: conv_state_out must be contiguous and non-null");
    }
    detail::causal_conv1d_prefill_split_launch(x, weight, conv_state_in, conv_state_out, out_query,
                                               out_key, out_value, stream);
}

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream) {"""),
    ("""void require_metadata_accessible(const Tensor& metadata, const char* label) {""",
     """void require_split_out_shape(const Tensor& x, const Tensor& out_query, const Tensor& out_key,
                             const Tensor& out_value) {
    const auto rank2 = [](const Tensor& t) { return t.ne[0] > 0 && t.ne[2] == 1 && t.ne[3] == 1; };
    if (!rank2(out_query) || !rank2(out_key) || !rank2(out_value)) {
        throw std::invalid_argument("causal_conv1d: split destinations must have shape [C,T]");
    }
    if (out_query.ne[1] != x.ne[1] || out_key.ne[1] != x.ne[1] || out_value.ne[1] != x.ne[1]) {
        throw std::invalid_argument("causal_conv1d: split destinations must have x's column count");
    }
    if (out_query.ne[0] + out_key.ne[0] + out_value.ne[0] != x.ne[0]) {
        throw std::invalid_argument("causal_conv1d: split destination rows must sum to C");
    }
}

void require_metadata_accessible(const Tensor& metadata, const char* label) {"""),
])

# --- 5. место вызова и рецепт рабочей области --------------------------------------------------
patch('src/targets/qwen3_6/impl/runtime/text_context_impl.h', [
    ("""        const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor qkv_c = conv.convolved;
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c, s);
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);""",
     """        Tensor qkv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qc, kc, vc, s);"""),
])

patch('src/targets/qwen3_6/impl/runtime/workspace_recipe.h', [
    ("""struct GdnPrefillConvRoots {
    Tensor projected;
    Tensor convolved;
};

template <class Config, class Allocator>
GdnPrefillConvRoots gdn_prefill_conv(Allocator& allocator, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, Config::convolution_dim, tokens),
        matrix(allocator, DType::BF16, Config::convolution_dim, tokens),
    };
}""",
     """// The convolution writes q/k/v directly into the projection roots, so the prefill path needs the
// projection destination only; the former `convolved` sibling existed solely to be sliced.
template <class Config, class Allocator>
Tensor gdn_prefill_conv(Allocator& allocator, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, Config::convolution_dim, tokens);
}"""),
])
print('готово')
