// Megakernel v0.3b harness: the COMPLETE 35B-A3B GDN layer at T=1, verbatim
// engine bodies, 48 layers:
//   norm      : rmsnorm2048(x) -> h                                  (1 slice)
//   gating    : g_beta[64] = Wg[64x2048] @ h                         (2 slices)
//   in+conv   : W8 12288x2048 @ h; rows<8192 -> 4-tap conv+silu ->
//               qc[2048]/kc[2048]/vc[4096] + in-place conv state;
//               rows>=8192 -> z[4096]                                (768 slices)
//   recurrent : gated delta net, 32 heads x 128x128 f32 state,
//               q/k L2-normalized, per-warp units                    (64 slices)
//   gatednorm : on = rmsnorm128(o) * silu(z)                         (1 slice)
//   out-proj  : x += W8 2048x4096 @ on                               (128 slices)
// DAG: gating || in+conv after norm; recurrent waits first 512 in-slices (conv
// rows) + gating; gatednorm waits recurrent + FULL in (z is the tail);
// norm(l+1) waits out + full in. Ref: 6 stream-ordered launches/layer (288).
// Gate: bitwise-identical outputs (same bodies, same partitioning, states reset
// between modes).
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

constexpr int kHidden     = 2048;
constexpr int kInRows     = 12288;
constexpr int kConvRows   = 8192;
constexpr int kValueDim   = 4096;   // 32 heads x 128
constexpr int kOutK       = 4096;
constexpr int kLayers     = 48;
constexpr int kRowSlice   = 16;
constexpr int kInSlices   = kInRows / kRowSlice;    // 768
constexpr int kHeadSlices = kConvRows / kRowSlice;  // 512 (q/k/v rows)
constexpr int kOutSlices  = kHidden / kRowSlice;    // 128
constexpr int kRecUnits   = 32 * 32;                // heads x (128/4) dv tiles
constexpr int kRecPerSl   = 16;                     // one warp per unit
constexpr int kRecSlices  = kRecUnits / kRecPerSl;  // 64
constexpr int kGatRows    = 64;
constexpr int kGatSlice   = 32;
constexpr float kEps      = 1e-6f;
constexpr float kGdnScale = 0.08838834764831845f;   // 1/sqrt(128)

// MoE (engine constants: 256 experts + shared, top-8, intermediate 512)
constexpr int kExperts    = 256;
constexpr int kRouterRows = kExperts + 1;
constexpr int kTopK       = 8;
constexpr int kMoeInter   = 512;
constexpr int kD1Slices   = (kRouterRows + 1) / 2;      // 129 (2 rows per CTA)
constexpr int kD3JBlock       = 16;   // consecutive j per slice, one warp per j
constexpr int kD3SharedSlices = kMoeInter / kD3JBlock;            // 32
constexpr int kD3RoutedSlices = 8 * kMoeInter / kD3JBlock;        // 256
constexpr int kD4RowSlice = 16;
constexpr int kD4Slices   = kHidden / kD4RowSlice;      // 256
constexpr int kCtr        = 27;   // 14 dep + 13 pop counters per layer (d3 split)

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

__global__ void fill_bf16_kernel(__nv_bfloat16* data, std::size_t n, std::uint32_t seed,
                                 float amplitude) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) { return; }
    std::uint32_t s = seed ^ static_cast<std::uint32_t>(i * 2654435761u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const float unit = static_cast<float>(s & 0xffffu) * (1.0f / 32768.0f) - 1.0f;
    data[i]          = __float2bfloat16_rn(amplitude * unit);
}

static std::int64_t pack_f32(float v) {
    int bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<std::int64_t>(bits);
}

int main() {
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const int n_sm = prop.multiProcessorCount;
    std::printf("device: %s, SMs=%d\n", prop.name, n_sm);

    // ---- buffers -------------------------------------------------------------
    __nv_bfloat16 *d_x, *d_h, *d_qc, *d_kc, *d_vc, *d_z, *d_o, *d_on, *d_g;
    CHECK(cudaMalloc(&d_x, kHidden * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_h, kHidden * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_qc, 2048 * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_kc, 2048 * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_vc, kValueDim * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_z, kValueDim * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_o, kValueDim * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_on, kValueDim * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_g, kGatRows * sizeof(__nv_bfloat16)));

    __nv_bfloat16* d_h2;
    float *d_scores, *d_alpha, *d_sscale, *d_act;
    float *d_ab, *d_gf, *d_betaf;
    int* d_ids;
    CHECK(cudaMalloc(&d_ab, kGatRows * sizeof(float)));
    CHECK(cudaMalloc(&d_gf, 32 * sizeof(float)));
    CHECK(cudaMalloc(&d_betaf, 32 * sizeof(float)));
    CHECK(cudaMalloc(&d_h2, kHidden * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&d_scores, kRouterRows * sizeof(float)));
    CHECK(cudaMalloc(&d_alpha, kTopK * sizeof(float)));
    CHECK(cudaMalloc(&d_sscale, sizeof(float)));
    CHECK(cudaMalloc(&d_act, (kTopK + 1) * kMoeInter * sizeof(float)));
    CHECK(cudaMalloc(&d_ids, kTopK * sizeof(int)));

    // Routed expert banks are SHARED across layers (per-layer routers pick
    // different top-8 sets, so the streamed addresses still differ per layer).
    const std::size_t rgu_rows  = static_cast<std::size_t>(kExperts) * 2 * kMoeInter;  // 262144
    const std::size_t rgu_codes = rgu_rows * (kHidden / 64) * 32;
    const std::size_t rgu_scal  = rgu_rows * (kHidden / 64);
    const std::size_t rd_rows   = static_cast<std::size_t>(kExperts) * kHidden;        // 524288
    const std::size_t rd_codes  = rd_rows * (kMoeInter / 64) * 32;
    const std::size_t rd_high   = rd_rows * (kMoeInter / 64) * 8;
    const std::size_t rd_scal   = rd_rows * (kMoeInter / 64);
    std::uint8_t *d_rgu_codes, *d_rd_codes, *d_rd_high;
    std::uint16_t *d_rgu_scales, *d_rd_scales;
    CHECK(cudaMalloc(&d_rgu_codes, rgu_codes));
    CHECK(cudaMalloc(&d_rgu_scales, rgu_scal * sizeof(std::uint16_t)));
    CHECK(cudaMalloc(&d_rd_codes, rd_codes));
    CHECK(cudaMalloc(&d_rd_high, rd_high));
    CHECK(cudaMalloc(&d_rd_scales, rd_scal * sizeof(std::uint16_t)));
    fill_codes_kernel<<<static_cast<unsigned>((rgu_codes + 255) / 256), 256>>>(d_rgu_codes,
                                                                               rgu_codes, 11u);
    fill_scales_kernel<<<static_cast<unsigned>((rgu_scal + 255) / 256), 256>>>(d_rgu_scales,
                                                                               rgu_scal, 12u);
    fill_codes_kernel<<<static_cast<unsigned>((rd_codes + 255) / 256), 256>>>(d_rd_codes,
                                                                              rd_codes, 13u);
    fill_codes_kernel<<<static_cast<unsigned>((rd_high + 255) / 256), 256>>>(d_rd_high, rd_high,
                                                                             14u);
    fill_scales_kernel<<<static_cast<unsigned>((rd_scal + 255) / 256), 256>>>(d_rd_scales,
                                                                              rd_scal, 15u);

    const std::size_t in_code_bytes   = static_cast<std::size_t>(kInRows) * kHidden;
    const std::size_t in_scale_count  = static_cast<std::size_t>(kInRows) * (kHidden / 32);
    const std::size_t out_code_bytes  = static_cast<std::size_t>(kHidden) * kOutK;
    const std::size_t out_scale_count = static_cast<std::size_t>(kHidden) * (kOutK / 32);
    const std::size_t rec_state_count = 32ull * 128 * 128;
    const std::size_t conv_state_count = 3ull * kConvRows;

    std::vector<__nv_bfloat16*> d_norm_w(kLayers), d_gw(kLayers), d_conv_w(kLayers),
        d_conv_state(kLayers), d_router(kLayers);
    std::vector<std::uint8_t*> d_in_codes(kLayers), d_out_codes(kLayers), d_sgu_codes(kLayers),
        d_sd_codes(kLayers);
    std::vector<std::uint16_t*> d_in_scales(kLayers), d_out_scales(kLayers),
        d_sgu_scales(kLayers), d_sd_scales(kLayers);
    std::vector<float*> d_rec_state(kLayers);
    std::vector<float*> d_alog(kLayers), d_dtb(kLayers);

    const std::size_t sgu_rows  = 2 * kMoeInter;                       // shared gate_up, K=2048
    const std::size_t sgu_codes = sgu_rows * kHidden;
    const std::size_t sgu_scal  = sgu_rows * (kHidden / 32);
    const std::size_t sd_rows   = static_cast<std::size_t>(kHidden);   // shared down, K=512
    const std::size_t sd_codes  = sd_rows * kMoeInter;
    const std::size_t sd_scal   = sd_rows * (kMoeInter / 32);

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (int l = 0; l < kLayers; ++l) {
        CHECK(cudaMalloc(&d_norm_w[l], kHidden * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_gw[l], kGatRows * kHidden * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_conv_w[l], 4 * kConvRows * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_conv_state[l], conv_state_count * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_rec_state[l], rec_state_count * sizeof(float)));
        CHECK(cudaMalloc(&d_alog[l], 32 * sizeof(float)));
        CHECK(cudaMalloc(&d_dtb[l], 32 * sizeof(float)));
        {
            std::vector<float> alog(32), dtb(32);
            for (int h = 0; h < 32; ++h) {
                alog[h] = -1.5f + 0.05f * dist(rng);   // exp(A_log) ~ 0.2 => alpha in (0,1)
                dtb[h]  = 0.3f * dist(rng);
            }
            CHECK(cudaMemcpy(d_alog[l], alog.data(), 32 * sizeof(float),
                             cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_dtb[l], dtb.data(), 32 * sizeof(float),
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
        std::vector<__nv_bfloat16> gw(kGatRows * kHidden);
        for (auto& e : gw) { e = __float2bfloat16_rn(0.02f * dist(rng)); }
        CHECK(cudaMemcpy(d_gw[l], gw.data(), gw.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
        std::vector<__nv_bfloat16> cw(4 * kConvRows);
        for (auto& e : cw) { e = __float2bfloat16_rn(0.2f * dist(rng)); }
        CHECK(cudaMemcpy(d_conv_w[l], cw.data(), cw.size() * sizeof(__nv_bfloat16),
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

        CHECK(cudaMalloc(&d_router[l], kRouterRows * kHidden * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&d_sgu_codes[l], sgu_codes));
        CHECK(cudaMalloc(&d_sgu_scales[l], sgu_scal * sizeof(std::uint16_t)));
        CHECK(cudaMalloc(&d_sd_codes[l], sd_codes));
        CHECK(cudaMalloc(&d_sd_scales[l], sd_scal * sizeof(std::uint16_t)));
        fill_bf16_kernel<<<static_cast<unsigned>((kRouterRows * kHidden + 255) / 256), 256>>>(
            d_router[l], kRouterRows * kHidden, seed ^ 0x27d4eb2fu, 0.03f);
        fill_codes_kernel<<<static_cast<unsigned>((sgu_codes + 255) / 256), 256>>>(
            d_sgu_codes[l], sgu_codes, seed ^ 0x165667b1u);
        fill_scales_kernel<<<static_cast<unsigned>((sgu_scal + 255) / 256), 256>>>(
            d_sgu_scales[l], sgu_scal, seed ^ 0xd3a2646cu);
        fill_codes_kernel<<<static_cast<unsigned>((sd_codes + 255) / 256), 256>>>(
            d_sd_codes[l], sd_codes, seed ^ 0xfd7046c5u);
        fill_scales_kernel<<<static_cast<unsigned>((sd_scal + 255) / 256), 256>>>(
            d_sd_scales[l], sd_scal, seed ^ 0xb55a4f09u);
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> x_init(kHidden);
    for (auto& e : x_init) { e = __float2bfloat16_rn(dist(rng)); }

    // ---- instruction builders -------------------------------------------------
    auto blank = [] {
        MkInstr instr{};
        instr.task_counter  = kMkNone;
        instr.slice_count   = 1;
        instr.done_counter  = kMkNone;
        instr.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { instr.wait_counter[w] = kMkNone; }
        return instr;
    };
    auto make_norm = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::RmsNorm2048;
        instr.ptr[0]  = d_x;
        instr.ptr[1]  = d_norm_w[l];
        instr.out[0]  = d_h;
        instr.dim[0]  = 1;
        instr.dim[1]  = pack_f32(kEps);
        return instr;
    };
    auto make_gating = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::Bf16Gemv;
        instr.ptr[0]  = d_h;
        instr.ptr[1]  = d_gw[l];
        instr.out[0]  = d_ab;
        instr.dim[0]  = 0;
        instr.dim[1]  = kGatSlice;
        instr.dim[2]  = kHidden;
        instr.dim[3]  = 1;   // fp32 out
        return instr;
    };
    auto make_gg = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::GdnGating;
        instr.ptr[0]  = d_ab;
        instr.ptr[1]  = d_ab + 32;
        instr.ptr[2]  = d_alog[l];
        instr.ptr[3]  = d_dtb[l];
        instr.out[0]  = d_gf;
        instr.out[1]  = d_betaf;
        instr.dim[0]  = 32;
        return instr;
    };
    auto make_inconv = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::W8DecodeConv;
        instr.ptr[0]  = d_h;
        instr.ptr[1]  = d_in_codes[l];
        instr.ptr[2]  = d_in_scales[l];
        instr.ptr[3]  = d_conv_w[l];
        instr.ptr[4]  = d_conv_state[l];
        instr.ptr[5]  = d_vc;
        instr.ptr[6]  = d_z;
        instr.out[0]  = d_qc;
        instr.out[1]  = d_kc;
        instr.dim[0]  = 0;
        instr.dim[1]  = kRowSlice;
        return instr;
    };
    auto make_rec = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::GdnRecurrent;
        instr.ptr[0]  = d_qc;
        instr.ptr[1]  = d_kc;
        instr.ptr[2]  = d_vc;
        instr.ptr[3]  = d_gf;
        instr.ptr[4]  = d_betaf;
        instr.ptr[5]  = nullptr;
        instr.ptr[6]  = nullptr;
        instr.out[0]  = d_o;
        instr.out[1]  = d_rec_state[l];
        instr.dim[0]  = 0;
        instr.dim[1]  = kRecPerSl;
        instr.dim[2]  = pack_f32(kGdnScale);
        instr.dim[3]  = 0;
        return instr;
    };
    auto make_gn = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::GatedNorm128;
        instr.ptr[0]  = d_o;
        instr.ptr[1]  = d_norm_w[l];
        instr.ptr[2]  = d_z;
        instr.out[0]  = d_on;
        instr.dim[0]  = 0;
        instr.dim[1]  = kValueDim / 128;
        instr.dim[2]  = pack_f32(kEps);
        return instr;
    };
    auto make_out = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::W8DecodeK;
        instr.ptr[0]  = d_on;
        instr.ptr[1]  = d_out_codes[l];
        instr.ptr[2]  = d_out_scales[l];
        instr.out[0]  = d_x;
        instr.dim[0]  = 0;
        instr.dim[1]  = kRowSlice;
        instr.dim[2]  = kOutK;
        instr.dim[3]  = 1;
        return instr;
    };
    auto make_norm2 = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::RmsNorm2048;
        instr.ptr[0]  = d_x;
        instr.ptr[1]  = d_norm_w[l];
        instr.out[0]  = d_h2;
        instr.dim[0]  = 1;
        instr.dim[1]  = pack_f32(kEps);
        return instr;
    };
    auto make_d1 = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::MoeD1;
        instr.ptr[0]  = d_h2;
        instr.ptr[1]  = d_router[l];
        instr.out[0]  = d_scores;
        instr.dim[0]  = 0;
        instr.dim[1]  = 2;
        return instr;
    };
    auto make_d2 = [&] {
        MkInstr instr = blank();
        instr.op      = MkOp::MoeD2;
        instr.ptr[0]  = d_scores;
        instr.ptr[1]  = d_sscale;
        instr.out[0]  = d_ids;
        instr.out[1]  = d_alpha;
        instr.dim[0]  = 0;
        instr.dim[1]  = 1;
        return instr;
    };
    auto make_d3 = [&](int l, bool shared_only) {
        MkInstr instr = blank();
        instr.op      = MkOp::MoeD3;
        instr.ptr[0]  = d_h2;
        instr.ptr[1]  = d_ids;
        instr.ptr[2]  = d_rgu_codes;
        instr.ptr[4]  = d_rgu_scales;
        instr.ptr[5]  = d_sgu_codes[l];
        instr.ptr[6]  = d_sgu_scales[l];
        instr.out[0]  = d_act;
        instr.dim[0]  = 0;
        instr.dim[1]  = kD3JBlock;
        instr.dim[3]  = shared_only ? 1 : 0;
        return instr;
    };
    auto make_d4 = [&](int l) {
        MkInstr instr = blank();
        instr.op      = MkOp::MoeD4;
        instr.ptr[0]  = d_ids;
        instr.ptr[1]  = d_alpha;
        instr.ptr[2]  = d_sscale;
        instr.ptr[3]  = d_act;
        instr.ptr[4]  = d_rd_codes;
        instr.ptr[5]  = d_rd_high;
        instr.ptr[6]  = d_rd_scales;
        instr.ptr[7]  = d_sd_codes[l];
        instr.out[1]  = d_sd_scales[l];
        instr.out[0]  = d_x;
        instr.dim[0]  = 0;
        instr.dim[1]  = kD4RowSlice;
        return instr;
    };

    // ---- tape (identical for every SM) ---------------------------------------
    // deps per layer: c_norm=+0, c_in=+1, c_in_head=+2, c_out=+3, c_gat=+4,
    // c_rec=+5, c_gn=+6, c_norm2=+7, c_d1=+8, c_d2=+9, c_d3=+10, c_d4=+11;
    // pops +12..+22 (norm,gat,in,rec,gn,out,norm2,d1,d2,d3,d4)
    std::vector<MkInstr> tape;
    for (int l = 0; l < kLayers; ++l) {
        MkInstr norm      = make_norm(l);
        norm.task_counter = kCtr * l + 13;
        norm.done_counter = kCtr * l;
        if (l > 0) {
            norm.wait_counter[0] = kCtr * (l - 1) + 11;
            norm.wait_target[0]  = kD4Slices;
            norm.wait_counter[1] = kCtr * (l - 1) + 1;
            norm.wait_target[1]  = kInSlices;
        }
        tape.push_back(norm);

        MkInstr gat         = make_gating(l);
        gat.task_counter    = kCtr * l + 14;
        gat.slice_count     = kGatRows / kGatSlice;
        gat.done_counter    = kCtr * l + 4;
        gat.wait_counter[0] = kCtr * l;
        gat.wait_target[0]  = 1;
        gat.dim[6] = 1;
        tape.push_back(gat);


        MkInstr in         = make_inconv(l);
        in.task_counter    = kCtr * l + 15;
        in.slice_count     = kInSlices;
        in.done_counter    = kCtr * l + 1;
        in.done2_counter   = kCtr * l + 2;
        in.done2_limit     = kHeadSlices;
        in.wait_counter[0] = kCtr * l;
        in.wait_target[0]  = 1;
        tape.push_back(in);

        MkInstr gg         = make_gg(l);
        gg.task_counter    = kCtr * l + 24;
        gg.done_counter    = kCtr * l + 12;
        gg.wait_counter[0] = kCtr * l + 4;
        gg.wait_target[0]  = kGatRows / kGatSlice;
        tape.push_back(gg);

        MkInstr rec         = make_rec(l);
        rec.task_counter    = kCtr * l + 16;
        rec.slice_count     = kRecSlices;
        rec.done_counter    = kCtr * l + 5;
        rec.wait_counter[0] = kCtr * l + 2;
        rec.wait_target[0]  = kHeadSlices;
        rec.wait_counter[1] = kCtr * l + 12;
        rec.wait_target[1]  = 1;
        rec.dim[6] = 1;
        tape.push_back(rec);

        MkInstr gn         = make_gn(l);
        gn.task_counter    = kCtr * l + 17;
        gn.done_counter    = kCtr * l + 6;
        gn.wait_counter[0] = kCtr * l + 5;
        gn.wait_target[0]  = kRecSlices;
        gn.wait_counter[1] = kCtr * l + 1;
        gn.wait_target[1]  = kInSlices;
        tape.push_back(gn);

        MkInstr out_i         = make_out(l);
        out_i.task_counter    = kCtr * l + 18;
        out_i.slice_count     = kOutSlices;
        out_i.done_counter    = kCtr * l + 3;
        out_i.wait_counter[0] = kCtr * l + 6;
        out_i.wait_target[0]  = 1;
        out_i.dim[6] = 1;
        tape.push_back(out_i);

        MkInstr norm2         = make_norm2(l);
        norm2.task_counter    = kCtr * l + 19;
        norm2.done_counter    = kCtr * l + 7;
        norm2.wait_counter[0] = kCtr * l + 3;
        norm2.wait_target[0]  = kOutSlices;
        tape.push_back(norm2);

        MkInstr d1         = make_d1(l);
        d1.task_counter    = kCtr * l + 20;
        d1.slice_count     = kD1Slices;
        d1.done_counter    = kCtr * l + 8;
        d1.wait_counter[0] = kCtr * l + 7;
        d1.wait_target[0]  = 1;
        d1.dim[6] = 1;
        tape.push_back(d1);

        MkInstr d2         = make_d2();
        d2.task_counter    = kCtr * l + 21;
        d2.done_counter    = kCtr * l + 9;
        d2.wait_counter[0] = kCtr * l + 8;
        d2.wait_target[0]  = kD1Slices;
        tape.push_back(d2);

        MkInstr d3s        = make_d3(l, true);
        d3s.task_counter   = kCtr * l + 25;
        d3s.slice_count    = kD3SharedSlices;
        d3s.done_counter   = kCtr * l + 24;
        d3s.wait_counter[0] = kCtr * l + 7;
        d3s.wait_target[0]  = 1;
        d3s.dim[6] = 1;
        tape.push_back(d3s);

        MkInstr d3         = make_d3(l, false);
        d3.task_counter    = kCtr * l + 22;
        d3.slice_count     = kD3RoutedSlices;
        d3.done_counter    = kCtr * l + 10;
        d3.wait_counter[0] = kCtr * l + 9;
        d3.wait_target[0]  = 1;
        d3.dim[6] = 1;
        tape.push_back(d3);

        MkInstr d4         = make_d4(l);
        d4.task_counter    = kCtr * l + 23;
        d4.slice_count     = kD4Slices;
        d4.done_counter    = kCtr * l + 11;
        d4.wait_counter[0] = kCtr * l + 10;
        d4.wait_target[0]  = kD3RoutedSlices;
        d4.wait_counter[1] = kCtr * l + 24;
        d4.wait_target[1]  = kD3SharedSlices;
        d4.wait_counter[2] = kCtr * l + 9;
        d4.wait_target[2]  = 1;
        d4.dim[6] = 1;
        tape.push_back(d4);
    }

    MkInstr* d_tape = nullptr;
    CHECK(cudaMalloc(&d_tape, tape.size() * sizeof(MkInstr)));
    CHECK(cudaMemcpy(d_tape, tape.data(), tape.size() * sizeof(MkInstr),
                     cudaMemcpyHostToDevice));
    std::vector<MkStream> streams(static_cast<std::size_t>(2 * n_sm),
                                  MkStream{d_tape, static_cast<std::uint32_t>(tape.size())});
    MkStream* d_streams = nullptr;
    CHECK(cudaMalloc(&d_streams, streams.size() * sizeof(MkStream)));
    CHECK(cudaMemcpy(d_streams, streams.data(), streams.size() * sizeof(MkStream),
                     cudaMemcpyHostToDevice));

    std::uint32_t* d_counters = nullptr;
    const int counter_count   = kCtr * kLayers;
    CHECK(cudaMalloc(&d_counters, counter_count * sizeof(std::uint32_t)));

    // ---- run helpers ----------------------------------------------------------
    auto reset_all = [&]() {
        CHECK(cudaMemcpy(d_x, x_init.data(), x_init.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
        for (int l = 0; l < kLayers; ++l) {
            CHECK(cudaMemset(d_conv_state[l], 0, conv_state_count * sizeof(__nv_bfloat16)));
            CHECK(cudaMemset(d_rec_state[l], 0, rec_state_count * sizeof(float)));
        }
    };
    auto run_ref = [&](cudaStream_t stream) {
        for (int l = 0; l < kLayers; ++l) {
            mk_ref_rmsnorm_kernel<<<1, kMkThreads, 0, stream>>>(make_norm(l));
            mk_ref_generic_kernel<<<kGatRows / kGatSlice, kMkThreads, 0, stream>>>(
                make_gating(l));
            mk_ref_generic_kernel<<<1, kMkThreads, 0, stream>>>(make_gg(l));
            mk_ref_generic_kernel<<<kInSlices, kMkThreads, 0, stream>>>(make_inconv(l));
            mk_ref_generic_kernel<<<kRecSlices, kMkThreads, 0, stream>>>(make_rec(l));
            mk_ref_generic_kernel<<<1, kMkThreads, 0, stream>>>(make_gn(l));
            mk_ref_w8_decode_kernel<<<kOutSlices, kMkThreads, 0, stream>>>(make_out(l));
            mk_ref_rmsnorm_kernel<<<1, kMkThreads, 0, stream>>>(make_norm2(l));
            mk_ref_generic_kernel<<<kD1Slices, kMkThreads, 0, stream>>>(make_d1(l));
            mk_ref_generic_kernel<<<1, kMkThreads, 0, stream>>>(make_d2());
            mk_ref_generic_kernel<<<kD3SharedSlices, kMkThreads, 0, stream>>>(make_d3(l, true));
            mk_ref_generic_kernel<<<kD3RoutedSlices, kMkThreads, 0, stream>>>(make_d3(l, false));
            mk_ref_generic_kernel<<<kD4Slices, kMkThreads, 0, stream>>>(make_d4(l));
        }
    };
    auto run_mk = [&](cudaStream_t stream, int prefetch) {
        CHECK(cudaMemsetAsync(d_counters, 0, counter_count * sizeof(std::uint32_t), stream));
        mk_interpreter_kernel<<<n_sm, kMkThreads, 0, stream>>>(d_streams, d_counters, prefetch,
                                                               nullptr, nullptr, nullptr,
                                                               nullptr, nullptr, 0);
    };

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // ---- correctness: bitwise identical outputs -------------------------------
    std::vector<__nv_bfloat16> out_ref(kHidden), out_mk(kHidden);
    reset_all();
    run_ref(stream);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out_ref.data(), d_x, kHidden * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    reset_all();
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

    reset_all();
    for (int i = 0; i < warmup; ++i) { run_ref(stream); }
    CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) { run_ref(stream); }
    CHECK(cudaEventRecord(t1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms_ref = 0.0f;
    CHECK(cudaEventElapsedTime(&ms_ref, t0, t1));

    auto time_mk = [&](int prefetch) {
        reset_all();
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

    // ---- per-class attribution (one instrumented pass) ------------------------
    unsigned long long *d_wait_ns, *d_exec_ns;
    CHECK(cudaMalloc(&d_wait_ns, tape.size() * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_exec_ns, tape.size() * sizeof(unsigned long long)));
    CHECK(cudaMemset(d_wait_ns, 0, tape.size() * sizeof(unsigned long long)));
    CHECK(cudaMemset(d_exec_ns, 0, tape.size() * sizeof(unsigned long long)));
    reset_all();
    CHECK(cudaMemsetAsync(d_counters, 0, counter_count * sizeof(std::uint32_t), stream));
    mk_interpreter_kernel<<<n_sm, kMkThreads, 0, stream>>>(d_streams, d_counters, 1, d_wait_ns,
                                                           d_exec_ns, nullptr, nullptr,
                                                           nullptr, 0);
    CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned long long> wait_h(tape.size()), exec_h(tape.size());
    CHECK(cudaMemcpy(wait_h.data(), d_wait_ns, tape.size() * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(exec_h.data(), d_exec_ns, tape.size() * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));
    const char* class_names[12] = {"norm",  "gating", "gg",     "in+conv", "recurrent", "gnorm",
                                   "out",   "norm2",  "moe_d1", "moe_d2",  "moe_d3",    "moe_d4"};
    double wait_sum[12] = {}, exec_sum[12] = {};
    for (std::size_t i = 0; i < tape.size(); ++i) {
        wait_sum[i % 12] += static_cast<double>(wait_h[i]);
        exec_sum[i % 12] += static_cast<double>(exec_h[i]);
    }
    std::printf("per-class totals, summed over 170 CTAs x 48 layers (us at 2.4GHz):\n");
    for (int c = 0; c < 12; ++c) {
        std::printf("  %-10s wait %9.0f  exec %9.0f\n", class_names[c], wait_sum[c] / 2400.0,
                    exec_sum[c] / 2400.0);
    }

    const double us_ref = 1e3 * ms_ref / iters;
    const double us_mk0 = 1e3 * ms_mk0 / iters;
    const double us_mk1 = 1e3 * ms_mk1 / iters;
    const double moe_bytes =
        2.0 * kRouterRows * kHidden +
        static_cast<double>(kTopK) * 2 * kMoeInter * ((kHidden / 64) * 34.0) +
        static_cast<double>(sgu_codes) + 2.0 * sgu_scal +
        static_cast<double>(kTopK) * kHidden * ((kMoeInter / 64) * 42.0) +
        static_cast<double>(sd_codes) + 2.0 * sd_scal;
    const double gb =
        kLayers *
        (static_cast<double>(in_code_bytes) + out_code_bytes +
         2.0 * (in_scale_count + out_scale_count) + 2.0 * kGatRows * kHidden +
         2.0 * 4 * kConvRows + 2.0 * 2 * conv_state_count + 8.0 * rec_state_count +
         moe_bytes) /
        1e9;
    const int boundaries = kLayers * 12;
    std::printf("traffic: %.2f GB/pass\n", gb);
    std::printf("ref (per-op kernels):     %8.2f us/pass  (%.0f GB/s)\n", us_ref,
                gb / us_ref * 1e6);
    std::printf("mk  (no prefetch):        %8.2f us/pass  (%.0f GB/s)  %+.1f%% vs ref\n", us_mk0,
                gb / us_mk0 * 1e6, 100.0 * (us_mk0 / us_ref - 1.0));
    std::printf("mk  (L2 weight prefetch): %8.2f us/pass  (%.0f GB/s)  %+.1f%% vs ref\n", us_mk1,
                gb / us_mk1 * 1e6, 100.0 * (us_mk1 / us_ref - 1.0));
    std::printf("boundary cost: %.0f ns per boundary (%d boundaries)\n",
                1e3 * (us_ref - us_mk1) / boundaries, boundaries);
    std::printf("MK_V03C_DONE\n");
    return 0;
}
