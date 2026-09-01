// Правда ли f16-аккумулятор вдвое быстрее f32-аккумулятора на этой карте.
// Три формы одной и той же m16n8k16, все в регистрах, цепочка из 32 MMA в теле цикла —
// то же правило приёмки, что и у mmapeak.
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCKS 680
#define THREADS 256
#define ITERS 20000

__global__ void peak_bf16_acc32(float* sink) {
    float c[4] = {0.f, 1.f, 2.f, 3.f};
    unsigned a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 4, b1 = a0 + 5;
#pragma unroll 1
    for (int i = 0; i < ITERS; ++i) {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (c[0] == 1234.5f) { sink[0] = c[0] + c[1] + c[2] + c[3]; }
}

__global__ void peak_f16_acc32(float* sink) {
    float c[4] = {0.f, 1.f, 2.f, 3.f};
    unsigned a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 4, b1 = a0 + 5;
#pragma unroll 1
    for (int i = 0; i < ITERS; ++i) {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (c[0] == 1234.5f) { sink[0] = c[0] + c[1] + c[2] + c[3]; }
}

__global__ void peak_f16_acc16(float* sink) {
    unsigned d0 = 0u, d1 = 1u;
    unsigned a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 4, b1 = a0 + 5;
#pragma unroll 1
    for (int i = 0; i < ITERS; ++i) {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                         "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                         : "+r"(d0), "+r"(d1)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (d0 == 0x1234u) { sink[0] = __int_as_float((int)(d0 + d1)); }
}

template <class K> static void run(const char* name, K kernel, float* sink, double base) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    kernel<<<BLOCKS, THREADS>>>(sink);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    kernel<<<BLOCKS, THREADS>>>(sink);
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    // m16n8k16 = 2*16*8*16 flop na warp-instrukciyu
    const double warps = (double)BLOCKS * (THREADS / 32);
    const double flop  = warps * (double)ITERS * 32.0 * 2.0 * 16 * 8 * 16;
    const double tops  = flop / (ms * 1e-3) / 1e12;
    printf("%-34s %8.3f ms  %8.1f TOP/s", name, ms, tops);
    if (base > 0.0) { printf("   x%.3f", tops / base); }
    printf("\n");
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  cuda: %s\n", cudaGetErrorString(e)); }
}

int main() {
    float* sink;
    cudaMalloc(&sink, sizeof(float));
    printf("m16n8k16, tri formy akkumulyatora, %d blokov x %d nitey\n\n", BLOCKS, THREADS);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    peak_bf16_acc32<<<BLOCKS, THREADS>>>(sink);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    peak_bf16_acc32<<<BLOCKS, THREADS>>>(sink);
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double warps = (double)BLOCKS * (THREADS / 32);
    const double flop  = warps * (double)ITERS * 32.0 * 2.0 * 16 * 8 * 16;
    const double base  = flop / (ms * 1e-3) / 1e12;

    run("bf16 operandy, f32 akkumulyator", peak_bf16_acc32, sink, base);
    run("f16 operandy,  f32 akkumulyator", peak_f16_acc32, sink, base);
    run("f16 operandy,  F16 AKKUMULYATOR", peak_f16_acc16, sink, base);
    printf("\npovtor:\n");
    run("bf16 operandy, f32 akkumulyator", peak_bf16_acc32, sink, base);
    run("f16 operandy,  F16 AKKUMULYATOR", peak_f16_acc16, sink, base);
    return 0;
}
