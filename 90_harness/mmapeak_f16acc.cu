// Piki MMA na etoy karte. Proshlaya versiya vrala: sink pisalsya pri threadIdx.x == 1024,
// chego pri 256 nityah ne byvaet, i ves schet ushel v mertvyy kod. Teper zapis bezuslovnaya,
// a oshibka zapuska proveryaetsya.
#include <cstdio>
#include <cuda_runtime.h>

#define ACCS 8

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                     \
            printf("  CUDA FAIL %s: %s\n", #x, cudaGetErrorString(e));                             \
            return -1.0;                                                                            \
        }                                                                                          \
    } while (0)

template <int Mode>
__global__ void peak_kernel(float* __restrict__ sink, int iters) {
    unsigned a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
    unsigned b[2] = {0x3c003c00u, 0x3c003c00u};
    unsigned c[ACCS][4] = {};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int k = 0; k < ACCS; ++k) {
            if constexpr (Mode == 0) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else if constexpr (Mode == 1) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else if constexpr (Mode == 2) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                             : "+r"(c[k][0]), "+r"(c[k][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
    }
    float s = 0;
#pragma unroll
    for (int k = 0; k < ACCS; ++k) { s += __int_as_float(c[k][0]) + __int_as_float(c[k][1]); }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int Mode>
double measure(const char* name, double ops_per_mma, int blocks, int threads, int iters) {
    float* sink = nullptr;
    CK(cudaMalloc(&sink, sizeof(float) * blocks * threads));
    peak_kernel<Mode><<<blocks, threads>>>(sink, 64);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    peak_kernel<Mode><<<blocks, threads>>>(sink, iters);
    cudaEventRecord(t1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    const double warps = double(blocks) * (threads / 32);
    const double total = warps * double(iters) * ACCS * ops_per_mma;
    const double tops = total / (ms / 1000.0) / 1e12;
    printf("  %-44s %8.1f   (%.3f ms)\n", name, tops, ms);
    cudaFree(sink);
    return tops;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    const int blocks = p.multiProcessorCount * 4;
    const int threads = 256;
    const int iters = 20000;
    printf("piki MMA: %d blokov po %d nitey na %d SM, %d iteraciy po %d nakopiteley\n", blocks,
           threads, p.multiProcessorCount, iters, ACCS);
    double bf = measure<0>("m16n8k16 bf16 -> f32", 16.0 * 8 * 16 * 2, blocks, threads, iters);
    double f32 = measure<1>("m16n8k16 f16  -> f32", 16.0 * 8 * 16 * 2, blocks, threads, iters);
    double f16 = measure<2>("m16n8k16 f16  -> f16", 16.0 * 8 * 16 * 2, blocks, threads, iters);
    double s8 = measure<3>("m16n8k32 s8   -> s32", 16.0 * 8 * 32 * 2, blocks, threads, iters);
    if (f32 > 0 && f16 > 0) { printf("\n  f16-nakoplenie / f32-nakoplenie: x%.3f\n", f16 / f32); }
    if (bf > 0 && s8 > 0) { printf("  s8 / bf16: x%.3f\n", s8 / bf); }
    return 0;
}
