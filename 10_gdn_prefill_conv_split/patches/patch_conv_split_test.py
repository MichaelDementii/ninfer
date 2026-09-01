#!/usr/bin/env python3
"""П7, часть 3: тест новой публичной формы.

Два утверждения в одном случае. Первое — операторное: каждый из трёх приёмников сверяется с тем же
полным FP64-оракулом по своему диапазону каналов. Второе — механизменное: та же вызовная точка,
но со слитым приёмником, даёт **побитово** те же биты, потому что меняется только адрес записи.
"""
import sys

sys.path.insert(0, '/root/exp')
from g2port import patch  # noqa: E402

patch('tests/ops/test_causal_conv1d_silu.cpp', [
    ("""// Continuation prefill reads a selected committed slot and publishes the new running state to slot
// 0.""",
     """// The split-destination entry writes the same values into three tensors instead of one. Both
// claims are checked here: each destination against its own channel range of the same complete
// oracle, and the whole triple against the fused entry bit for bit, because the two forms differ
// only in the store address.
int split_case(std::int32_t C, std::int32_t T, std::int32_t rows_q, std::int32_t rows_k,
               std::uint32_t seed) {
    const std::int32_t rows_v      = C - rows_q - rows_k;
    const LogicalInput input       = make_input(C, T, seed);
    const std::vector<float> state = make_state(C, seed + 2U);
    const OracleResult oracle      = causal_conv_oracle(input.x, input.weight, state, C, T, false);
    const std::vector<std::uint16_t> x_bits      = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits  = bf16_bits(state);
    const std::vector<std::uint16_t> final_bits  = bf16_bits(oracle.final_state);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_in(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_out(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer fused(x_bits.size() * sizeof(std::uint16_t));
    const std::array<std::int32_t, 3> rows{rows_q, rows_k, rows_v};
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> destinations;
    for (const std::int32_t r : rows) {
        destinations.push_back(std::make_unique<GuardedDeviceBuffer>(
            static_cast<std::size_t>(r) * static_cast<std::size_t>(T) * sizeof(std::uint16_t)));
        destinations.back()->fill(kOutputPoison);
    }
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state_in.copy_from_host(state_bits.data(), state_in.bytes());
    state_out.fill(0x5a);
    fused.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor ts_in(state_in.data(), DType::BF16, {C, 3});
    Tensor ts_out(state_out.data(), DType::BF16, {C, 3});
    Tensor tfused(fused.data(), DType::BF16, {C, T});
    Tensor tq(destinations[0]->data(), DType::BF16, {rows_q, T});
    Tensor tk(destinations[1]->data(), DType::BF16, {rows_k, T});
    Tensor tv(destinations[2]->data(), DType::BF16, {rows_v, T});

    ops::causal_conv1d_silu(tx, tw, ts_in, ts_out, tq, tk, tv, nullptr);
    ops::causal_conv1d_silu(tx, tw, ts_in, ts_out, tfused, nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu split C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " rows=" + std::to_string(rows_q) + "," +
                            std::to_string(rows_k) + "," + std::to_string(rows_v);
    const std::vector<std::uint16_t> fused_bits = from_device<std::uint16_t>(
        fused.data(), static_cast<std::size_t>(C) * static_cast<std::size_t>(T));

    int failures         = 0;
    std::int32_t base    = 0;
    const char* names[3] = {"query", "key", "value"};
    for (std::size_t d = 0; d < rows.size(); ++d) {
        const std::int32_t r = rows[d];
        std::vector<double> reference(static_cast<std::size_t>(r) * static_cast<std::size_t>(T));
        std::vector<std::uint16_t> fused_slice(reference.size());
        for (std::int32_t t = 0; t < T; ++t) {
            for (std::int32_t i = 0; i < r; ++i) {
                reference[offset(i, t, r)]   = oracle.output[offset(base + i, t, C)];
                fused_slice[offset(i, t, r)] = fused_bits[offset(base + i, t, C)];
            }
        }
        failures +=
            verify_output(tag + " " + names[d] + " against oracle",
                          from_device_bf16(destinations[d]->data(), reference.size()), reference);
        failures += verify_bits(tag + " " + names[d] + " equals fused entry",
                                destinations[d]->data(), fused_slice);
        failures += verify_buffer_guards(tag + " " + names[d], *destinations[d]);
        base += r;
    }
    failures += verify_bits(tag + " final state", state_out.data(), final_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_bits(tag + " initial state preserved", state_in.data(), state_bits);
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " state input", state_in);
    failures += verify_buffer_guards(tag + " state output", state_out);
    failures += verify_buffer_guards(tag + " fused", fused);
    return failures;
}

// Continuation prefill reads a selected committed slot and publishes the new running state to slot
// 0."""),

    ("""    failures += ordinary_case(kQwen35Channels, 257, StateCall::InPlaceEntry, 3257U);""",
     """    failures += ordinary_case(kQwen35Channels, 257, StateCall::InPlaceEntry, 3257U);

    // Split destinations. 2048/2048/6144 is the production GDN partition; the odd-row case leaves
    // the pair route and exercises the scalar one, and T=1 and T=3 cover the state-window prefix.
    for (const std::int32_t T : {1, 3, 17, 64, 65, 257}) {
        failures +=
            split_case(kQwen27Channels, T, 2048, 2048, 6000U + static_cast<std::uint32_t>(T));
    }
    failures += split_case(kQwen35Channels, 33, 1025, 1025, 6033U);"""),
])
print('тест дополнен')
