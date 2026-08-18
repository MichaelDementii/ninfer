// Unit exactness: engine recurrent_bf16_direct (T=1) and w8_rowsplit_gemm_simt
// residual out-proj (T=1) vs mk GdnRecurrent / W8DecodeK bodies.
#define NINFER_MK_ENGINE 1
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"
#include "ops/megakernel/mk_core.cuh"
#include "ops/megakernel/mk_instr.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace ninfer;
using namespace ninfer::ops;
#define CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void mk_rec_driver(mk::MkInstr instr) {
    mk::MkInstr local = instr;
    local.dim[0] = instr.dim[0] + blockIdx.x * instr.dim[1];
    mk::mk_body_gdn_recurrent(local);
}
__global__ void mk_out_driver(mk::MkInstr instr) {
    mk::MkInstr local = instr;
    local.dim[0] = instr.dim[0] + blockIdx.x * instr.dim[1];
    mk::mk_body_w8_decode<4096>(local);
}

int main() {
    srand(23);
    auto rb=[](){ return __float2bfloat16_rn((rand()/(float)RAND_MAX-0.5f)*2.0f); };

    // ================= REC =================
    {
        const int HQK=16, HV=32, D=128;
        std::vector<__nv_bfloat16> hq(HQK*D), hk(HQK*D), hv(HV*D);
        std::vector<float> hg(HV), hb(HV), hstate((size_t)HV*D*D);
        for (auto& x: hq) x=rb();
        for (auto& x: hk) x=rb();
        for (auto& x: hv) x=rb();
        for (auto& x: hg) x = -0.5f - 0.01f*(rand()%100);
        for (auto& x: hb) x = 0.01f*(rand()%100);
        for (auto& x: hstate) x = (rand()/(float)RAND_MAX-0.5f)*0.5f;
        __nv_bfloat16 *dq,*dk,*dv,*doutA,*doutB; float *dg,*db,*dstA,*dstB;
        CHECK(cudaMalloc(&dq,HQK*D*2)); CHECK(cudaMalloc(&dk,HQK*D*2)); CHECK(cudaMalloc(&dv,HV*D*2));
        CHECK(cudaMalloc(&doutA,HV*D*2)); CHECK(cudaMalloc(&doutB,HV*D*2));
        CHECK(cudaMalloc(&dg,HV*4)); CHECK(cudaMalloc(&db,HV*4));
        CHECK(cudaMalloc(&dstA,(size_t)HV*D*D*4)); CHECK(cudaMalloc(&dstB,(size_t)HV*D*D*4));
        CHECK(cudaMemcpy(dq,hq.data(),HQK*D*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dk,hk.data(),HQK*D*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dv,hv.data(),HV*D*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dg,hg.data(),HV*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(db,hb.data(),HV*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dstA,hstate.data(),(size_t)HV*D*D*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dstB,hstate.data(),(size_t)HV*D*D*4,cudaMemcpyHostToDevice));
        const dim3 grid(HV, 1, D/detail::gated_delta_net::kBlockDv);
        const dim3 block(32, detail::gated_delta_net::kNumWarps, 1);
        detail::gated_delta_net::recurrent_bf16_direct_kernel<true><<<grid, block>>>(
            dq, dk, dv, dg, db, dstA, dstA, doutA, 1, detail::gated_delta_net::head_map::of(HQK, HV), 0.75f);
        CHECK(cudaDeviceSynchronize());

        mk::MkInstr rc{};
        rc.op = mk::MkOp::GdnRecurrent;
        rc.ptr[0]=dq; rc.ptr[1]=dk; rc.ptr[2]=dv; rc.ptr[3]=dg; rc.ptr[4]=db;
        rc.ptr[5]=nullptr; rc.ptr[6]=nullptr;
        rc.out[0]=doutB; rc.out[1]=dstB;
        rc.dim[0]=0; rc.dim[1]=16; int sb; float sc=0.75f; memcpy(&sb,&sc,4); rc.dim[2]=sb;
        rc.dim[3]=0; rc.dim[7]=0;
        mk_rec_driver<<<(HV*32)/16, mk::kMkThreads>>>(rc);
        CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> oa(HV*D), ob(HV*D);
        std::vector<float> sa((size_t)HV*D*D), sbv((size_t)HV*D*D);
        CHECK(cudaMemcpy(oa.data(),doutA,HV*D*2,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(ob.data(),doutB,HV*D*2,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(sa.data(),dstA,(size_t)HV*D*D*4,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(sbv.data(),dstB,(size_t)HV*D*D*4,cudaMemcpyDeviceToHost));
        int bad=0;
        for (int i=0;i<HV*D && bad<4;++i) if (memcmp(&oa[i],&ob[i],2)) {printf("rec out[%d] eng %04x mk %04x\n", i, *(unsigned short*)&oa[i], *(unsigned short*)&ob[i]); ++bad;}
        printf("rec out %s\n", bad?"MISMATCH":"BITEXACT");
        bad=0;
        for (size_t i=0;i<(size_t)HV*D*D && bad<4;++i) if (memcmp(&sa[i],&sbv[i],4)) {printf("rec state[%zu] eng %.9g mk %.9g\n", i, sa[i], sbv[i]); ++bad;}
        printf("rec state %s\n", bad?"MISMATCH":"BITEXACT");
    }

    // ================= OUT-PROJ =================
    {
        const int R=2048, K=4096;
        std::vector<__nv_bfloat16> hx(K), hres(R);
        std::vector<uint8_t> hcodes((size_t)R*K);
        std::vector<uint16_t> hscales((size_t)R*(K/32));
        for (auto& v: hx) v=rb();
        for (auto& v: hres) v=rb();
        for (auto& v: hcodes) v=(uint8_t)(rand()&0xff);
        for (auto& v: hscales) { __half h=__float2half(0.001f+0.002f*(rand()%100)); memcpy(&v,&h,2); }
        __nv_bfloat16 *dx,*dresA,*dresB; uint8_t* dcodes; uint16_t* dscales;
        CHECK(cudaMalloc(&dx,K*2)); CHECK(cudaMalloc(&dresA,R*2)); CHECK(cudaMalloc(&dresB,R*2));
        CHECK(cudaMalloc(&dcodes,(size_t)R*K)); CHECK(cudaMalloc(&dscales,(size_t)R*(K/32)*2));
        CHECK(cudaMemcpy(dx,hx.data(),K*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dresA,hres.data(),R*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dresB,hres.data(),R*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dcodes,hcodes.data(),(size_t)R*K,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dscales,hscales.data(),(size_t)R*(K/32)*2,cudaMemcpyHostToDevice));

        const detail::W8ContiguousOutput output{dresA, R};
        const dim3 grid(R/8, 1, 1);
        detail::w8_rowsplit_gemm_simt_kernel<detail::W8RowSplitSimtSchedule, 4, 8, 2, false,
                                             detail::W8Epilogue::Residual>
            <<<grid, 8*32>>>(dx, dcodes, (const uint8_t*)dscales, output, R, K, 1, K, K/1024);
        CHECK(cudaDeviceSynchronize());

        mk::MkInstr ot{};
        ot.op = mk::MkOp::W8DecodeK;
        ot.ptr[0]=dx; ot.ptr[1]=dcodes; ot.ptr[2]=dscales; ot.out[0]=dresB;
        ot.dim[0]=0; ot.dim[1]=16; ot.dim[2]=K; ot.dim[3]=1;
        mk_out_driver<<<R/16, mk::kMkThreads>>>(ot);
        CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> ra(R), rbv(R);
        CHECK(cudaMemcpy(ra.data(),dresA,R*2,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rbv.data(),dresB,R*2,cudaMemcpyDeviceToHost));
        int bad=0;
        for (int i=0;i<R && bad<6;++i) if (memcmp(&ra[i],&rbv[i],2)) {printf("out[%d] eng %04x mk %04x\n", i, *(unsigned short*)&ra[i], *(unsigned short*)&rbv[i]); ++bad;}
        printf("out-proj %s\n", bad?"MISMATCH":"BITEXACT");
    }
    return 0;
}
