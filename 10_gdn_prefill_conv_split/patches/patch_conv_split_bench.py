#!/usr/bin/env python3
"""П7, часть 2: операторный бенч со своим контролем в той же таблице.

Строка `split-control` — ровно то, что маршрут делает сегодня: свёртка в слитый приёмник плюс три
`extract_bf16_columns`. Строка `split` — новая форма с тремя приёмниками. Обеим объявлен один и тот
же компульсорный трафик (4n + 20C), поэтому колонка ГБ/с у них — это чистое отношение времён, а не
разная модель байт. Форма 1:1:3 повторяет продовую раскладку GDN: [C/5, C/5, 3C/5].
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

patch('bench/ops/causal_conv1d_silu_bench.cu', [
    ("""#include "ninfer/ops/causal_conv1d_silu.h"
#include "core/device.h\"""",
     """#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/scatter.h" // extract_bf16_columns, for the split control
#include "core/device.h\""""),

    ("""    bool decode   = false;
    bool prefill  = false;
    bool distinct = false;
    bool snapshot = false;
};""",
     """    bool decode   = false;
    bool prefill  = false;
    bool distinct = false;
    bool snapshot = false;
    bool split    = false;
};"""),

    ("""void run_decode(const Options& options) {""",
     """// Three-destination prefill form against the sequence it replaces. The control is the production
// path as written today: convolve into a fused destination, then slice it with three copies. Both
// rows declare the same compulsory traffic as the fused form, so the reported GB/s is a pure time
// ratio between them rather than two different byte models.
void run_split(const Options& options) {
    const std::int32_t C      = options.channels;
    const std::int32_t T      = options.tokens;
    const std::int32_t rows_q = C / 5;
    const std::int32_t rows_k = C / 5;
    const std::int32_t rows_v = C - rows_q - rows_k;

    const std::size_t n       = static_cast<std::size_t>(C) * T;
    const std::size_t state_n = static_cast<std::size_t>(C) * 3u;

    DeviceBuffer x         = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight    = make_varied_bf16(static_cast<std::size_t>(C) * 4u, 0x87654321U);
    DeviceBuffer state_in  = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer state_out = make_zeros(state_n * 2u);
    DeviceBuffer fused     = make_zeros(n * 2u);
    DeviceBuffer out_q     = make_zeros(static_cast<std::size_t>(rows_q) * T * 2u);
    DeviceBuffer out_k     = make_zeros(static_cast<std::size_t>(rows_k) * T * 2u);
    DeviceBuffer out_v     = make_zeros(static_cast<std::size_t>(rows_v) * T * 2u);

    Tensor tx(x.p, DType::BF16, {C, T});
    Tensor tw(weight.p, DType::BF16, {C, 4});
    Tensor tin(state_in.p, DType::BF16, {C, 3});
    Tensor tout_state(state_out.p, DType::BF16, {C, 3});
    Tensor tfused(fused.p, DType::BF16, {C, T});
    Tensor tq(out_q.p, DType::BF16, {rows_q, T});
    Tensor tk(out_k.p, DType::BF16, {rows_k, T});
    Tensor tv(out_v.p, DType::BF16, {rows_v, T});

    const double bytes = 4.0 * static_cast<double>(n) + 20.0 * C;

    const Result control = bench_loop(
        [&](cudaStream_t s) {
            ops::causal_conv1d_silu(tx, tw, tin, tout_state, tfused, s);
            ops::extract_bf16_columns(tfused, 0, tq, s);
            ops::extract_bf16_columns(tfused, rows_q, tk, s);
            ops::extract_bf16_columns(tfused, rows_q + rows_k, tv, s);
        },
        bytes);
    const std::string control_tag = shape_tag("split-control", C, T);
    print_result(control_tag.c_str(), control);

    const Result r = bench_loop(
        [&](cudaStream_t s) { ops::causal_conv1d_silu(tx, tw, tin, tout_state, tq, tk, tv, s); },
        bytes);
    const std::string tag = shape_tag("split", C, T);
    print_result(tag.c_str(), r);
}

void run_decode(const Options& options) {"""),

    ("""        } else if (!std::strcmp(argv[i], "--snapshot")) {
            options.snapshot = true;""",
     """        } else if (!std::strcmp(argv[i], "--snapshot")) {
            options.snapshot = true;
        } else if (!std::strcmp(argv[i], "--split")) {
            options.split = true;"""),

    ("""    if (!options.decode && !options.prefill && !options.distinct && !options.snapshot) {
        options.decode = options.prefill = true;
    }""",
     """    if (!options.decode && !options.prefill && !options.distinct && !options.snapshot &&
        !options.split) {
        options.decode = options.prefill = true;
    }
    if (options.split && (options.channels % 10 != 0)) { return false; }"""),

    ("""                 "usage: %s [--decode] [--prefill] [--distinct] [--snapshot] \"""",
     """                 "usage: %s [--decode] [--prefill] [--distinct] [--snapshot] [--split] \""""),

    ("""    if (options.snapshot) run_snapshot(options);
    return 0;""",
     """    if (options.snapshot) run_snapshot(options);
    if (options.split) run_split(options);
    return 0;"""),
])
print('бенч дополнен')
