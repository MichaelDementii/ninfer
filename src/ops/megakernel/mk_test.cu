// Megakernel v0.2a harness: real 35B-A3B dense shapes, verbatim engine bodies.
//
// Per pseudo-layer (uniform, 48 layers):
//   norm    : rmsnorm2048(x) -> h                       (1 instr, grid=1 in ref)
//   inproj  : u[12288] = W8_l @ h, K=2048               (768 slices x 16 rows, verbatim
//                                                        w8_k2048_decode body)
//   outproj : x[2048] += W8b_l @ u[0:4096], K=4096      (128 slices x 16 rows)
// ~34 MB of weights per layer, ~1.63 GB per pass — the same DRAM streaming regime as
// the real decode step, so per-layer weights never fit L2 and boundary costs show at
// true scale. Ref mode: 3 kernel launches/layer (144/pass). MK mode: one persistent
// interpreter launch + counter sync. Gate: bitwise-identical outputs.
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

constexpr int kHidden    = 2048;
constexpr int kInRows    = 12288;  // gdn qkvz projection rows
constexpr int kOutK      = 4096;   // out-projection reduction width
constexpr int kLayers    = 48;
constexpr int kRowSlice  = 16;
constexpr int kInSlices  = kInRows / kRowSlice;   // 768
constexpr int kOutSlices = kHidden / kRowSlice;   // 128
constexpr float kEps     = 1e-6f;

// ---- device-side fill (1.6 GB of weights: host generation would be too slow) ----
__global__ void fill_codes_kernel(std::uint8_t* data, std::size_t n, std::uint32_t seed) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) { return; }
    std::uint32_t s = seed ^ static_cast<std::uint32_t>(i * 2654435761u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    data[i] = static_cast<std::uint8_t>(s);
}

__global__ void fill_scales_kernel(std::uint16_t* data, std::size_t n, std::uint32_t seed) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) { return; }
    std::uint32_t s = seed ^ static_cast<std::uint32_t>(i * 2246822519u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const float value = 0.001f + static_cast<float>(s & 1023u) * 1e-5f;
    data[i] = __half_as_ushort(__float2half(value));
}

static std::int64_t pack_eps(float eps) {
    int bits;
    std::memcpy(&bits, &eps, sizeof(bits));
    return static_cast<std::int64_t>(bits);
}

int main() {
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const int n_sm = prop.multiProcessorCount;
    std::printf("device: %s, SMs=%d\n", prop.name, n_sm);

    // ---- buffers -------------------------------------------------------------
    __nv_bfloat16 *d_x = nullptr, *d_h = nullptr, *d_u = nullptr;
    __nv_bfloat16 *d_g = nullptr, *d_gn = nullptr, *d_sg = nullptr;
    CHECK(cudaMalloc(&d_x, kHidden * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_h, kHidden * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_u, kInRows * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_g, 64 * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_gn, kOutK * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_sg, kOutK * sizeof(__nv_bfloat16)));

    const std::size_t in_code_bytes   = static_cast<std::size_t>(kInRows) * kHidden;
    const std::size_t in_scale_count  = static_cast<std::size_t>(kInRows) * (kHidden / 32);
    const std::size_t out_code_bytes  = static_cast<std::size_t>(kHidden) * kOutK;
    const std::size_t out_scale_count = static_cast<std::size_t>(kHidden) * (kOutK / 32);

    std::vector<__nv_bfloat16*> d_norm_w(kLayers), d_gw(kLayers);
    std::vector<std::uint8_t*> d_in_codes(kLayers), d_out_codes(kLayers);
    std::vector<std::uint16_t*> d_in_scales(kLayers), d_out_scales(kLayers);

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (int l = 0; l < kLayers; ++l) {
        CHECK(cudaMalloc(&d_norm_w[l], kHidden * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_gw[l], 64 * kHidden * sizeof(__nv_bfloat16)));
        {
            std::vector<__nv_bfloat16> gw(64 * kHidden);
            for (auto& e : gw) { e = __float2bfloat16_rn(0.05f * dist(rng)); }
            CHECK(cudaMemcpy(d_gw[l], gw.data(), gw.size() * sizeof(__nv_bfloat16),
                             cudaMemcpyHostToDevice));
        }
        CHECK(cudaMalloc(&d_in_codes[l], in_code_bytes));
        CHECK(cudaMalloc(&d_in_scales[l], in_scale_count * sizeof(std::uint16_t)));
        CHECK(cudaMalloc(&d_out_codes[l], out_code_bytes));
        CHECK(cudaMalloc(&d_out_scales[l], out_scale_count * sizeof(std::uint16_t)));

        std::vector<__nv_bfloat16> nw(kHidden);
        for (auto& e : nw) { e = __float2bfloat16_rn(dist(rng)); }
        CHECK(cudaMemcpy(d_norm_w[l], nw.data(), nw.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));

        const std::uint32_t seed = 77u + static_cast<std::uint32_t>(l);
        fill_codes_kernel<<<static_cast<unsigned>((in_code_bytes + 255) / 256), 256>>>(
            d_in_codes[l], in_code_bytes, seed);
        fill_scales_kernel<<<static_cast<unsigned>((in_scale_count + 255) / 256), 256>>>(
            d_in_scales[l], in_scale_count, seed ^ 0x9e3779b9u);
        fill_codes_kernel<<<static_cast<unsigned>((out_code_bytes + 255) / 256), 256>>>(
            d_out_codes[l], out_code_bytes, seed ^ 0x85ebca6bu);
        fill_scales_kernel<<<static_cast<unsigned>((out_scale_count + 255) / 256), 256>>>(
            d_out_scales[l], out_scale_count, seed ^ 0xc2b2ae35u);
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> x_init(kHidden);
    for (auto& e : x_init) { e = __float2bfloat16_rn(dist(rng)); }

    // ---- instruction builders -------------------------------------------------
    auto make_norm = [&](int l) {
        MkInstr instr{};
        instr.op            = MkOp::RmsNorm2048;
        instr.task_counter  = kMkNone;
        instr.slice_count   = 1;
        instr.done_counter  = kMkNone;
        instr.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { instr.wait_counter[w] = kMkNone; }
        instr.ptr[0] = d_x;
        instr.ptr[1] = d_norm_w[l];
        instr.out[0] = d_h;
        instr.dim[0] = 1;
        instr.dim[1] = pack_eps(kEps);
        return instr;
    };
    auto make_decode = [&](const void* x, const std::uint8_t* codes, const std::uint16_t* scales,
                           __nv_bfloat16* dst, std::int64_t row0, std::int64_t rows,
                           std::int64_t k, bool residual) {
        MkInstr instr{};
        instr.op            = MkOp::W8DecodeK;
        instr.task_counter  = kMkNone;
        instr.slice_count   = 1;
        instr.done_counter  = kMkNone;
        instr.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { instr.wait_counter[w] = kMkNone; }
        instr.ptr[0] = x;
        instr.ptr[1] = codes;
        instr.ptr[2] = scales;
        instr.out[0] = dst;
        instr.dim[0] = row0;
        instr.dim[1] = rows;
        instr.dim[2] = k;
        instr.dim[3] = residual ? 1 : 0;
        return instr;
    };

    // ---- tape (identical for every SM: 6 task classes per layer) --------------
    // Engine-like granularity: the small control chain (gating -> gated norm ->
    // sigmoid) runs in PARALLEL with the wide in-proj inside the megakernel,
    // joining before out-proj; the per-op reference is forced to serialize all
    // six launches — exactly the structural difference vs the real decode graph.
    // counters per layer l (13 each): deps c_norm=13l+0, c_in=+1, c_in_head=+2,
    // c_out=+3, c_gat=+4, c_gn=+5, c_sg=+6; pops +7..+12 (norm,in,out,gat,gn,sg).
    // gated norm reads u[0:8192] (x head + z region) -> in-proj posts done2 for
    // its first 512 slices. norm(l+1) still waits the FULL c_in(l).
    constexpr int kHeadSlices = (2 * kOutK) / kRowSlice;   // 512: u[0:8192]
    constexpr int kGatRows    = 64;
    constexpr int kGatSlice   = 32;
    constexpr int kCtr        = 13;
    std::vector<MkInstr> tape;
    for (int l = 0; l < kLayers; ++l) {
        MkInstr norm       = make_norm(l);
        norm.task_counter  = kCtr * l + 7;
        norm.done_counter  = kCtr * l;
        if (l > 0) {
            norm.wait_counter[0] = kCtr * (l - 1) + 3;
            norm.wait_target[0]  = kOutSlices;
            norm.wait_counter[1] = kCtr * (l - 1) + 1;
            norm.wait_target[1]  = kInSlices;
        }
        tape.push_back(norm);

        MkInstr gat{};
        gat.op            = MkOp::Bf16Gemv;
        gat.task_counter  = kCtr * l + 10;
        gat.slice_count   = kGatRows / kGatSlice;
        gat.done_counter  = kCtr * l + 4;
        gat.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { gat.wait_counter[w] = kMkNone; }
        gat.ptr[0]          = d_h;
        gat.ptr[1]          = d_gw[l];
        gat.out[0]          = d_g;
        gat.dim[0]          = 0;
        gat.dim[1]          = kGatSlice;
        gat.dim[2]          = kHidden;
        gat.wait_counter[0] = kCtr * l;
        gat.wait_target[0]  = 1;
        tape.push_back(gat);

        MkInstr in = make_decode(d_h, d_in_codes[l], d_in_scales[l], d_u, 0, kRowSlice,
                                 kHidden, false);
        in.task_counter    = kCtr * l + 8;
        in.slice_count     = kInSlices;
        in.done_counter    = kCtr * l + 1;
        in.done2_counter   = kCtr * l + 2;
        in.done2_limit     = kHeadSlices;
        in.wait_counter[0] = kCtr * l;
        in.wait_target[0]  = 1;
        tape.push_back(in);

        MkInstr gn{};
        gn.op            = MkOp::GatedNorm128;
        gn.task_counter  = kCtr * l + 11;
        gn.slice_count   = 1;
        gn.done_counter  = kCtr * l + 5;
        gn.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { gn.wait_counter[w] = kMkNone; }
        gn.ptr[0]          = d_u;
        gn.ptr[1]          = d_norm_w[l];
        gn.ptr[2]          = d_u + kOutK;
        gn.out[0]          = d_gn;
        gn.dim[0]          = 0;
        gn.dim[1]          = kOutK / 128;   // 32 rows of d=128
        gn.dim[2]          = pack_eps(kEps);
        gn.wait_counter[0] = kCtr * l + 2;
        gn.wait_target[0]  = kHeadSlices;
        gn.wait_counter[1] = kCtr * l + 4;
        gn.wait_target[1]  = kGatRows / kGatSlice;
        tape.push_back(gn);

        MkInstr sg{};
        sg.op            = MkOp::SigmoidMul;
        sg.task_counter  = kCtr * l + 12;
        sg.slice_count   = 1;
        sg.done_counter  = kCtr * l + 6;
        sg.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { sg.wait_counter[w] = kMkNone; }
        sg.ptr[0]          = d_gn;
        sg.ptr[1]          = d_g;
        sg.out[0]          = d_sg;
        sg.dim[0]          = 0;
        sg.dim[1]          = kOutK;
        sg.wait_counter[0] = kCtr * l + 5;
        sg.wait_target[0]  = 1;
        tape.push_back(sg);

        MkInstr out_i = make_decode(d_sg, d_out_codes[l], d_out_scales[l], d_x, 0, kRowSlice,
                                    kOutK, true);
        out_i.task_counter    = kCtr * l + 9;
        out_i.slice_count     = kOutSlices;
        out_i.done_counter    = kCtr * l + 3;
        out_i.wait_counter[0] = kCtr * l + 6;
        out_i.wait_target[0]  = 1;
        tape.push_back(out_i);
    }

    MkInstr* d_tape = nullptr;
    CHECK(cudaMalloc(&d_tape, tape.size() * sizeof(MkInstr)));
    CHECK(cudaMemcpy(d_tape, tape.data(), tape.size() * sizeof(MkInstr),
                     cudaMemcpyHostToDevice));
    std::vector<MkStream> streams(static_cast<std::size_t>(n_sm),
                                  MkStream{d_tape, static_cast<std::uint32_t>(tape.size())});
    MkStream* d_streams = nullptr;
    CHECK(cudaMalloc(&d_streams, streams.size() * sizeof(MkStream)));
    CHECK(cudaMemcpy(d_streams, streams.data(), streams.size() * sizeof(MkStream),
                     cudaMemcpyHostToDevice));

    std::uint32_t* d_counters = nullptr;
    const int counter_count   = kCtr * kLayers;
    CHECK(cudaMalloc(&d_counters, counter_count * sizeof(std::uint32_t)));

    // ---- run helpers ----------------------------------------------------------
    auto reset_x = [&]() {
        CHECK(cudaMemcpy(d_x, x_init.data(), x_init.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
    };
    auto run_ref = [&](cudaStream_t stream) {
        for (int l = 0; l < kLayers; ++l) {
            mk_ref_rmsnorm_kernel<<<1, kMkThreads, 0, stream>>>(make_norm(l));
            {
                MkInstr gat{};
                gat.op     = MkOp::Bf16Gemv;
                gat.ptr[0] = d_h;
                gat.ptr[1] = d_gw[l];
                gat.out[0] = d_g;
                gat.dim[0] = 0;
                gat.dim[1] = kGatSlice;
                gat.dim[2] = kHidden;
                mk_ref_generic_kernel<<<kGatRows / kGatSlice, kMkThreads, 0, stream>>>(gat);
            }
            mk_ref_w8_decode_kernel<<<kInSlices, kMkThreads, 0, stream>>>(
                make_decode(d_h, d_in_codes[l], d_in_scales[l], d_u, 0, kRowSlice, kHidden,
                            false));
            {
                MkInstr gn{};
                gn.op     = MkOp::GatedNorm128;
                gn.ptr[0] = d_u;
                gn.ptr[1] = d_norm_w[l];
                gn.ptr[2] = d_u + kOutK;
                gn.out[0] = d_gn;
                gn.dim[0] = 0;
                gn.dim[1] = kOutK / 128;
                gn.dim[2] = pack_eps(kEps);
                mk_ref_generic_kernel<<<1, kMkThreads, 0, stream>>>(gn);
                MkInstr sg{};
                sg.op     = MkOp::SigmoidMul;
                sg.ptr[0] = d_gn;
                sg.ptr[1] = d_g;
                sg.out[0] = d_sg;
                sg.dim[0] = 0;
                sg.dim[1] = kOutK;
                mk_ref_generic_kernel<<<1, kMkThreads, 0, stream>>>(sg);
            }
            mk_ref_w8_decode_kernel<<<kOutSlices, kMkThreads, 0, stream>>>(
                make_decode(d_sg, d_out_codes[l], d_out_scales[l], d_x, 0, kRowSlice, kOutK,
                            true));
        }
    };
    auto run_mk = [&](cudaStream_t stream, int prefetch) {
        CHECK(cudaMemsetAsync(d_counters, 0, counter_count * sizeof(std::uint32_t), stream));
        mk_interpreter_kernel<<<n_sm, kMkThreads, 0, stream>>>(d_streams, d_counters, prefetch);
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
    run_mk(stream, 1);
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
    const int warmup = 10, iters = 100;
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

    auto time_mk = [&](int prefetch) {
        for (int i = 0; i < warmup; ++i) { run_mk(stream, prefetch); }
        CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < iters; ++i) { run_mk(stream, prefetch); }
        CHECK(cudaEventRecord(t1, stream));
        CHECK(cudaStreamSynchronize(stream));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        return ms;
    };
    const float ms_mk0 = time_mk(0);
    const float ms_mk1 = time_mk(1);

    const double us_ref = 1e3 * ms_ref / iters;
    const double us_mk0 = 1e3 * ms_mk0 / iters;
    const double us_mk1 = 1e3 * ms_mk1 / iters;
    const double gb     = kLayers * (static_cast<double>(in_code_bytes) + out_code_bytes +
                                 2.0 * (in_scale_count + out_scale_count) +
                                 2.0 * kGatRows * kHidden) /
                      1e9;
    const int boundaries = kLayers * 6;
    std::printf("traffic: %.2f GB/pass\n", gb);
    std::printf("ref (per-op kernels):     %8.2f us/pass  (%.0f GB/s)\n", us_ref,
                gb / us_ref * 1e6);
    std::printf("mk  (no prefetch):        %8.2f us/pass  (%.0f GB/s)  %+.1f%% vs ref\n", us_mk0,
                gb / us_mk0 * 1e6, 100.0 * (us_mk0 / us_ref - 1.0));
    std::printf("mk  (L2 weight prefetch): %8.2f us/pass  (%.0f GB/s)  %+.1f%% vs ref\n", us_mk1,
                gb / us_mk1 * 1e6, 100.0 * (us_mk1 / us_ref - 1.0));
    std::printf("boundary cost: %.0f ns per boundary (%d boundaries)\n",
                1e3 * (us_ref - us_mk0) / boundaries, boundaries);
    std::printf("MK_V02A_DONE\n");
    return 0;
}
