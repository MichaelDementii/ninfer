// Megakernel v0.1 harness: a 48-layer dense pseudo-chain (rmsnorm -> W8 gemv+residual)
// executed two ways with IDENTICAL bodies and partitioning:
//   ref: one kernel launch per op (2 launches/layer, stream-ordered)
//   mk : one persistent interpreter launch walking per-SM tapes with counter sync
// Gate: outputs must match bit-for-bit. Metric: microseconds per chain pass;
// the difference isolates pure kernel-boundary overhead on this GPU.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120a mk_test.cu -o mk_test

#include "mk_core.cuh"
#include "mk_instr.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ninfer::ops::mk;

#define CHECK(call)                                                                     \
    do {                                                                                \
        cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
                         __FILE__, __LINE__);                                           \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

constexpr int kHidden   = 2048;
constexpr int kLayers   = 48;
constexpr int kRowSlice = 16;                       // rows per gemv instruction (one CTA)
constexpr int kSlices   = kHidden / kRowSlice;      // 128 gemv instructions per layer
constexpr float kEps    = 1e-6f;

static std::int64_t pack_eps(float eps) {
    int bits;
    std::memcpy(&bits, &eps, sizeof(bits));
    return static_cast<std::int64_t>(bits);
}

int main() {
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    const int n_sm = prop.multiProcessorCount;

    // ---- data ----------------------------------------------------------------
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    auto host_bf16 = [&](std::size_t n) {
        std::vector<__nv_bfloat16> v(n);
        for (auto& e : v) { e = __float2bfloat16_rn(dist(rng)); }
        return v;
    };

    const std::size_t code_bytes  = static_cast<std::size_t>(kHidden) * kHidden;
    const std::size_t group_count = static_cast<std::size_t>(kHidden) * (kHidden / 32);

    __nv_bfloat16 *d_x = nullptr, *d_h = nullptr;
    CHECK(cudaMalloc(&d_x, kHidden * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_h, kHidden * sizeof(__nv_bfloat16)));

    std::vector<__nv_bfloat16*> d_norm_w(kLayers);
    std::vector<std::int8_t*> d_codes(kLayers);
    std::vector<std::uint16_t*> d_scales(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        CHECK(cudaMalloc(&d_norm_w[l], kHidden * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_codes[l], code_bytes));
        CHECK(cudaMalloc(&d_scales[l], group_count * sizeof(std::uint16_t)));

        auto nw = host_bf16(kHidden);
        CHECK(cudaMemcpy(d_norm_w[l], nw.data(), nw.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
        std::vector<std::int8_t> codes(code_bytes);
        for (auto& c : codes) { c = static_cast<std::int8_t>((rng() % 255) - 127); }
        CHECK(cudaMemcpy(d_codes[l], codes.data(), codes.size(), cudaMemcpyHostToDevice));
        std::vector<std::uint16_t> scales(group_count);
        for (auto& s : scales) {
            const float value = 0.002f + 0.002f * dist(rng) * dist(rng);
            s                 = __half_as_ushort(__float2half(std::fabs(value)));
        }
        CHECK(cudaMemcpy(d_scales[l], scales.data(), scales.size() * sizeof(std::uint16_t),
                         cudaMemcpyHostToDevice));
    }
    const auto x_init = host_bf16(kHidden);

    // ---- instruction builders -------------------------------------------------
    auto make_norm = [&](int l) {
        MkInstr instr{};
        instr.op           = MkOp::RmsNorm2048;
        instr.done_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { instr.wait_counter[w] = kMkNone; }
        instr.ptr[0] = d_x;
        instr.ptr[1] = d_norm_w[l];
        instr.out[0] = d_h;
        instr.dim[0] = 1;
        instr.dim[1] = pack_eps(kEps);
        return instr;
    };
    auto make_gemv = [&](int l, std::int64_t row0, std::int64_t rows) {
        MkInstr instr{};
        instr.op           = MkOp::W8GemvResidual;
        instr.done_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { instr.wait_counter[w] = kMkNone; }
        instr.ptr[0] = d_h;
        instr.ptr[1] = d_codes[l];
        instr.ptr[2] = d_scales[l];
        instr.out[0] = d_x;
        instr.dim[0] = row0;
        instr.dim[1] = rows;
        instr.dim[2] = kHidden;
        return instr;
    };

    // ---- tapes ---------------------------------------------------------------
    // counters: 2 per layer; c_norm(l)=2l (target 1), c_gemv(l)=2l+1 (target kSlices)
    std::vector<std::vector<MkInstr>> tapes(static_cast<std::size_t>(n_sm));
    for (int l = 0; l < kLayers; ++l) {
        MkInstr norm       = make_norm(l);
        norm.done_counter  = 2 * l;
        if (l > 0) {
            norm.wait_counter[0] = 2 * (l - 1) + 1;
            norm.wait_target[0]  = kSlices;
        }
        tapes[0].push_back(norm);
        for (int s = 0; s < kSlices; ++s) {
            MkInstr gemv         = make_gemv(l, static_cast<std::int64_t>(s) * kRowSlice,
                                             kRowSlice);
            gemv.done_counter    = 2 * l + 1;
            gemv.wait_counter[0] = 2 * l;
            gemv.wait_target[0]  = 1;
            tapes[static_cast<std::size_t>(s % n_sm)].push_back(gemv);
        }
    }

    std::vector<MkInstr> flat;
    std::vector<MkStream> streams(static_cast<std::size_t>(n_sm));
    for (int s = 0; s < n_sm; ++s) {
        streams[static_cast<std::size_t>(s)].count =
            static_cast<std::uint32_t>(tapes[static_cast<std::size_t>(s)].size());
        streams[static_cast<std::size_t>(s)].tape =
            reinterpret_cast<const MkInstr*>(flat.size() * sizeof(MkInstr));  // offset, fixed below
        flat.insert(flat.end(), tapes[static_cast<std::size_t>(s)].begin(),
                    tapes[static_cast<std::size_t>(s)].end());
    }
    MkInstr* d_tape = nullptr;
    CHECK(cudaMalloc(&d_tape, flat.size() * sizeof(MkInstr)));
    CHECK(cudaMemcpy(d_tape, flat.data(), flat.size() * sizeof(MkInstr),
                     cudaMemcpyHostToDevice));
    for (auto& s : streams) {
        s.tape = reinterpret_cast<const MkInstr*>(
            reinterpret_cast<const char*>(d_tape) +
            reinterpret_cast<std::uintptr_t>(s.tape));
    }
    MkStream* d_streams = nullptr;
    CHECK(cudaMalloc(&d_streams, streams.size() * sizeof(MkStream)));
    CHECK(cudaMemcpy(d_streams, streams.data(), streams.size() * sizeof(MkStream),
                     cudaMemcpyHostToDevice));

    std::uint32_t* d_counters = nullptr;
    const int counter_count   = 2 * kLayers;
    CHECK(cudaMalloc(&d_counters, counter_count * sizeof(std::uint32_t)));

    // ---- run helpers ----------------------------------------------------------
    auto reset_x = [&]() {
        CHECK(cudaMemcpy(d_x, x_init.data(), x_init.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
    };
    auto run_ref = [&](cudaStream_t stream) {
        for (int l = 0; l < kLayers; ++l) {
            mk_ref_rmsnorm_kernel<<<1, kMkThreads, 0, stream>>>(make_norm(l));
            mk_ref_w8_gemv_kernel<<<kSlices, kMkThreads, 0, stream>>>(
                make_gemv(l, 0, kRowSlice));
        }
    };
    auto run_mk = [&](cudaStream_t stream) {
        CHECK(cudaMemsetAsync(d_counters, 0, counter_count * sizeof(std::uint32_t), stream));
        mk_interpreter_kernel<<<n_sm, kMkThreads, 0, stream>>>(d_streams, d_counters);
    };

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // ---- correctness: bitwise identical outputs -------------------------------
    std::vector<__nv_bfloat16> out_ref(kHidden), out_mk(kHidden);
    reset_x();
    run_ref(stream);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(out_ref.data(), d_x, kHidden * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    reset_x();
    run_mk(stream);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out_mk.data(), d_x, kHidden * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < kHidden; ++i) {
        if (std::memcmp(&out_ref[i], &out_mk[i], sizeof(__nv_bfloat16)) != 0) { ++mismatches; }
    }
    float max_abs = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
        max_abs = std::max(max_abs, std::fabs(__bfloat162float(out_mk[i])));
    }
    std::printf("correctness: %s (%d/%d mismatched, |out|max=%g)\n",
                mismatches == 0 ? "BITEXACT" : "MISMATCH", mismatches, kHidden, max_abs);
    if (mismatches != 0) { return 1; }

    // ---- timing ---------------------------------------------------------------
    const int warmup = 30, iters = 300;
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < warmup; ++i) { run_ref(stream); }
    CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) { run_ref(stream); }
    CHECK(cudaEventRecord(t1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms_ref = 0.0f;
    CHECK(cudaEventElapsedTime(&ms_ref, t0, t1));

    for (int i = 0; i < warmup; ++i) { run_mk(stream); }
    CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) { run_mk(stream); }
    CHECK(cudaEventRecord(t1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms_mk = 0.0f;
    CHECK(cudaEventElapsedTime(&ms_mk, t0, t1));

    const double us_ref = 1e3 * ms_ref / iters;
    const double us_mk  = 1e3 * ms_mk / iters;
    const int boundaries = kLayers * 2;
    std::printf("ref (per-op kernels): %8.2f us/pass\n", us_ref);
    std::printf("mk  (megakernel):     %8.2f us/pass\n", us_mk);
    std::printf("delta: %+.1f%%  (%.0f ns per eliminated boundary, %d boundaries)\n",
                100.0 * (us_mk / us_ref - 1.0), 1e3 * (us_ref - us_mk) / boundaries,
                boundaries);
    std::printf("MK_V01_DONE\n");
    return 0;
}
