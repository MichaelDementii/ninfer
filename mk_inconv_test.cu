// Unit exactness: engine w8_k2048_decode+conv (T=1) vs mk W8DecodeConv body.
#define NINFER_MK_ENGINE 1
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/w8/w8_k2048_decode.cuh"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/megakernel/mk_core.cuh"
#include "ops/megakernel/mk_instr.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
using namespace ninfer;
using namespace ninfer::ops;
#define CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

struct EngOutput {   // mirror of W8SplitOutput2<8192,4096> usage (unused with epilogue)
    __nv_bfloat16* a; __nv_bfloat16* b;
};

struct EngEpi {
    detail::GdnConvEpilogue<detail::SnapshotHistoryPublish> conv;
    __nv_bfloat16* z;
    template <class IgnoredOutput>
    __device__ __forceinline__ void operator()(const IgnoredOutput&, std::int32_t, std::int32_t row,
                                               float accumulator) const {
        if (row < 8192) {
            const float projected[1]{accumulator};
            conv.store(row, projected);
        } else {
            z[row - 8192] = __float2bfloat16_rn(accumulator);
        }
    }
};

__global__ void mk_inconv_driver(mk::MkInstr instr) {
    __shared__ mk::MkShared shared;
    mk::MkInstr local = instr;
    local.dim[0] = instr.dim[0] + blockIdx.x * instr.dim[1];
    mk::mk_body_w8_decode_conv(local);
}

int main() {
    const int R = 12288, K = 2048, CH = 8192;
    std::vector<__nv_bfloat16> hx(K), hconvw(4*CH), hstate(3*CH);
    std::vector<uint8_t> hcodes((size_t)R*K);
    std::vector<uint16_t> hscales((size_t)R*(K/32));
    srand(11);
    auto rb=[](){ return __float2bfloat16_rn((rand()/(float)RAND_MAX-0.5f)*2.0f); };
    for (auto& v: hx) v=rb();
    for (auto& v: hconvw) v=rb();
    for (auto& v: hstate) v=rb();
    for (auto& v: hcodes) v=(uint8_t)(rand()&0xff);
    for (auto& v: hscales) { __half h=__float2half(0.001f+0.002f*(rand()%100)); memcpy(&v,&h,2); }

    __nv_bfloat16 *dx,*dconvw,*dstateA,*dstateB,*dqA,*dkA,*dvA,*dzA,*dqB,*dkB,*dvB,*dzB;
    uint8_t *dcodes; uint16_t *dscales; int32_t *dslot;
    CHECK(cudaMalloc(&dx,K*2)); CHECK(cudaMalloc(&dconvw,4*CH*2));
    CHECK(cudaMalloc(&dstateA,3*CH*2)); CHECK(cudaMalloc(&dstateB,3*CH*2));
    CHECK(cudaMalloc(&dcodes,(size_t)R*K)); CHECK(cudaMalloc(&dscales,(size_t)R*(K/32)*2));
    CHECK(cudaMalloc(&dqA,2048*2)); CHECK(cudaMalloc(&dkA,2048*2)); CHECK(cudaMalloc(&dvA,4096*2)); CHECK(cudaMalloc(&dzA,4096*2));
    CHECK(cudaMalloc(&dqB,2048*2)); CHECK(cudaMalloc(&dkB,2048*2)); CHECK(cudaMalloc(&dvB,4096*2)); CHECK(cudaMalloc(&dzB,4096*2));
    CHECK(cudaMalloc(&dslot,4));
    CHECK(cudaMemcpy(dx,hx.data(),K*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dconvw,hconvw.data(),4*CH*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstateA,hstate.data(),3*CH*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dstateB,hstate.data(),3*CH*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dcodes,hcodes.data(),(size_t)R*K,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dscales,hscales.data(),(size_t)R*(K/32)*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dslot,0,4));

    // engine
    EngOutput ignored{dqA,dzA};
    EngEpi epi{
        { dconvw, dstateA, dslot, nullptr, dqA, dkA, dvA, CH, 2048, 2048, 4096, 0, 1, 0,
          detail::SnapshotHistoryPublish{dstateA, dslot, CH} },
        dzA,
    };
    detail::w8_k2048_decode_kernel<12288, 8, EngOutput, EngEpi>
        <<<12288/8, 8*32>>>(dx, dcodes, (const uint8_t*)dscales, ignored, epi);
    CHECK(cudaDeviceSynchronize());

    // mk body
    mk::MkInstr in{};
    in.op = mk::MkOp::W8DecodeConv;
    in.ptr[0]=dx; in.ptr[1]=dcodes; in.ptr[2]=dscales; in.ptr[3]=dconvw; in.ptr[4]=dstateB;
    in.ptr[5]=dvB; in.ptr[6]=dzB; in.ptr[7]=dslot;
    in.out[0]=dqB; in.out[1]=dkB;
    in.dim[0]=0; in.dim[1]=32;
    mk_inconv_driver<<<12288/32, mk::kMkThreads>>>(in);
    CHECK(cudaDeviceSynchronize());

    auto cmp=[&](const char* name, __nv_bfloat16* a, __nv_bfloat16* b, int n){
        std::vector<__nv_bfloat16> ha(n), hb(n);
        CHECK(cudaMemcpy(ha.data(),a,n*2,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hb.data(),b,n*2,cudaMemcpyDeviceToHost));
        int bad=0;
        for(int i=0;i<n && bad<4;++i) if(memcmp(&ha[i],&hb[i],2)){printf("%s[%d] eng %04x mk %04x\n",name,i,*(unsigned short*)&ha[i],*(unsigned short*)&hb[i]);++bad;}
        printf("%s %s\n", name, bad?"MISMATCH":"BITEXACT");
    };
    cmp("q",dqA,dqB,2048); cmp("k",dkA,dkB,2048); cmp("v",dvA,dvB,4096); cmp("z",dzA,dzB,4096);
    cmp("state",dstateA,dstateB,3*CH);
    return 0;
}
