// Unit exactness test: engine fused norm+gating (split-32, t=1) vs mk FGA/FGB.
#define NINFER_MK_ENGINE 1
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh"
#include "ops/megakernel/mk_core.cuh"
#include "ops/megakernel/mk_instr.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace ninfer;
using namespace ninfer::ops;

#define CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void mk_fg_a_driver(mk::MkInstr instr) {
    __shared__ mk::MkShared shared;
    mk::MkInstr local = instr;
    local.dim[0] = instr.dim[0] + blockIdx.x * instr.dim[1];
    mk::mk_body_fused_gate_a(local, shared);
}
__global__ void mk_fg_b_driver(mk::MkInstr instr) {
    mk::mk_body_fused_gate_b(instr);
}

int main() {
    const int H = 2048, N = 32;
    std::vector<__nv_bfloat16> hx(H), hw(H), ha(N*H), hb(N*H);
    std::vector<float> halog(N), hdt(N);
    srand(7);
    auto rb = [](){ return __float2bfloat16_rn((rand()/(float)RAND_MAX - 0.5f) * 2.0f); };
    for (auto& v : hx) v = rb();
    for (auto& v : hw) v = rb();
    for (auto& v : ha) v = rb();
    for (auto& v : hb) v = rb();
    for (auto& v : halog) v = -1.5f + 0.01f*(rand()%100);
    for (auto& v : hdt) v = 0.02f*(rand()%100) - 1.0f;

    __nv_bfloat16 *dx, *dw, *da, *db, *dh_ref, *dh_mk;
    float *dalog, *ddt, *dws, *dg_ref, *dbe_ref, *dg_mk, *dbe_mk, *dp_mk;
    CHECK(cudaMalloc(&dx, H*2)); CHECK(cudaMalloc(&dw, H*2));
    CHECK(cudaMalloc(&da, N*H*2)); CHECK(cudaMalloc(&db, N*H*2));
    CHECK(cudaMalloc(&dh_ref, H*2)); CHECK(cudaMalloc(&dh_mk, H*2));
    CHECK(cudaMalloc(&dalog, N*4)); CHECK(cudaMalloc(&ddt, N*4));
    CHECK(cudaMalloc(&dws, 2080*4)); CHECK(cudaMalloc(&dp_mk, 2080*4));
    CHECK(cudaMalloc(&dg_ref, N*4)); CHECK(cudaMalloc(&dbe_ref, N*4));
    CHECK(cudaMalloc(&dg_mk, N*4)); CHECK(cudaMalloc(&dbe_mk, N*4));
    CHECK(cudaMemcpy(dx, hx.data(), H*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, hw.data(), H*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(da, ha.data(), N*H*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb.data(), N*H*2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dalog, halog.data(), N*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ddt, hdt.data(), N*4, cudaMemcpyHostToDevice));

    // ---- engine reference: cooperative split-32 launch, t = 1, Predicated
    using Geo = detail::Bf16Gdn35Geometry;
    constexpr int kSmem = detail::kBf16GdnSmemBytes<Geo::kBlockN>;
    auto kern = detail::bf16_gdn_gating_proj_gemm_mma_kernel<Geo, 32, false, 8, true, 6>;
    CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1, 2, 32); cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = kSmem;
    cudaLaunchAttribute coop{}; coop.id = cudaLaunchAttributeCooperative; coop.val.cooperative = 1;
    cfg.attrs = &coop; cfg.numAttrs = 1;
    const __nv_bfloat16* cx = dx; const __nv_bfloat16* cw = dw;
    const __nv_bfloat16* ca = da; const __nv_bfloat16* cb = db;
    const float* calog = dalog; const float* cdt = ddt;
    std::int32_t t1 = 1;
    CHECK(cudaLaunchKernelEx(&cfg, kern, cx, cw, dh_ref, 1e-6f, ca, cb, calog, cdt, dws, dg_ref, dbe_ref, t1));
    CHECK(cudaDeviceSynchronize());

    // ---- mk path: FGA 64 slices + FGB
    mk::MkInstr fga{};
    fga.op = mk::MkOp::FusedGateA;
    fga.ptr[0]=dx; fga.ptr[1]=dw; fga.ptr[2]=da; fga.ptr[3]=db; fga.out[0]=dp_mk;
    fga.dim[0]=0; fga.dim[1]=1; fga.slice_count=64;
    mk_fg_a_driver<<<64, mk::kMkThreads>>>(fga);
    CHECK(cudaDeviceSynchronize());
    mk::MkInstr fgb{};
    fgb.op = mk::MkOp::FusedGateB;
    fgb.ptr[0]=dx; fgb.ptr[1]=dw; fgb.ptr[2]=dp_mk; fgb.ptr[3]=dalog; fgb.ptr[4]=ddt; fgb.ptr[5]=dbe_mk;
    fgb.out[0]=dh_mk; fgb.out[1]=dg_mk;
    int ebits; float ef=1e-6f; memcpy(&ebits,&ef,4); fgb.dim[1]=ebits;
    mk_fg_b_driver<<<1, mk::kMkThreads>>>(fgb);
    CHECK(cudaDeviceSynchronize());

    // ---- bitwise compare
    std::vector<__nv_bfloat16> h_ref(H), h_mk(H);
    std::vector<float> g_ref(N), g_mk(N), be_ref(N), be_mk(N), p_mk(2080), p_ref(2080);
    CHECK(cudaMemcpy(h_ref.data(), dh_ref, H*2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_mk.data(), dh_mk, H*2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(g_ref.data(), dg_ref, N*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(g_mk.data(), dg_mk, N*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(be_ref.data(), dbe_ref, N*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(be_mk.data(), dbe_mk, N*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p_ref.data(), dws, 2080*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p_mk.data(), dp_mk, 2080*4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 2080 && bad < 5; ++i) {
        if (memcmp(&p_ref[i], &p_mk[i], 4)) { printf("partial[%d] ref %.9g mk %.9g\n", i, p_ref[i], p_mk[i]); ++bad; }
    }
    printf("partials %s\n", bad ? "MISMATCH" : "BITEXACT");
    bad = 0;
    for (int i = 0; i < H && bad < 5; ++i) {
        if (memcmp(&h_ref[i], &h_mk[i], 2)) { printf("h[%d] ref %04x mk %04x\n", i, *(unsigned short*)&h_ref[i], *(unsigned short*)&h_mk[i]); ++bad; }
    }
    printf("h %s\n", bad ? "MISMATCH" : "BITEXACT");
    bad = 0;
    for (int i = 0; i < N && bad < 5; ++i) {
        if (memcmp(&g_ref[i], &g_mk[i], 4)) { printf("g[%d] ref %.9g mk %.9g\n", i, g_ref[i], g_mk[i]); ++bad; }
        if (memcmp(&be_ref[i], &be_mk[i], 4)) { printf("beta[%d] ref %.9g mk %.9g\n", i, be_ref[i], be_mk[i]); ++bad; }
    }
    printf("g/beta %s\n", bad ? "MISMATCH" : "BITEXACT");
    return 0;
}
