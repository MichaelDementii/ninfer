// Потолок тензорных ядер RTX 5090 (sm_120a) по трём типам, которые реально идут в префилле.
// Всё держится в регистрах: памяти в цикле нет, меряется чистая скорость выдачи MMA.
// bf16 и s8 служат проверкой харнеса: прошлая кампания намерила на этой же карте
// 256.7 TFLOP/s и 1008.6 TOPS, и если они воспроизводятся, числу по fp8 можно верить.
#include <cstdio>
#include <cuda_runtime.h>

#define ITERS 4096
#define CHAINS 8
#define WARPS 8
#define BLOCKS 680

__global__ __launch_bounds__(WARPS * 32) void bf16_peak(float* sink) {
    unsigned a0 = threadIdx.x, a1 = a0 ^ 1u, a2 = a0 ^ 2u, a3 = a0 ^ 3u;
    unsigned b0 = a0 ^ 5u, b1 = a0 ^ 7u;
    float c[CHAINS][4];
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) {
        c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.0f;
    }
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < CHAINS; ++i) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) { s += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (s == 1234.5f) { sink[0] = s; }
}

__global__ __launch_bounds__(WARPS * 32) void fp8_peak(float* sink) {
    unsigned a0 = threadIdx.x, a1 = a0 ^ 1u, a2 = a0 ^ 2u, a3 = a0 ^ 3u;
    unsigned b0 = a0 ^ 5u, b1 = a0 ^ 7u;
    float c[CHAINS][4];
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) {
        c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.0f;
    }
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < CHAINS; ++i) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) { s += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (s == 1234.5f) { sink[0] = s; }
}

__global__ __launch_bounds__(WARPS * 32) void s8_peak(float* sink) {
    unsigned a0 = threadIdx.x, a1 = a0 ^ 1u, a2 = a0 ^ 2u, a3 = a0 ^ 3u;
    unsigned b0 = a0 ^ 5u, b1 = a0 ^ 7u;
    int c[CHAINS][4];
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) {
        c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0;
    }
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < CHAINS; ++i) {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    int s = 0;
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) { s += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (s == 1234567) { sink[0] = 1.0f; }
}

__global__ __launch_bounds__(WARPS * 32) void nvfp4_peak(float* sink) {
    unsigned a0 = threadIdx.x, a1 = a0 ^ 1u, a2 = a0 ^ 2u, a3 = a0 ^ 3u;
    unsigned b0 = a0 ^ 5u, b1 = a0 ^ 7u;
    unsigned sfa = 0x3f3f3f3fu, sfb = 0x3f3f3f3fu;
    constexpr unsigned short kBlk = 0, kThr = 0;
    float c[CHAINS][4];
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) { c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.0f; }
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int i = 0; i < CHAINS; ++i) {
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                         "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                         "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                         : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                           "r"(sfa), "h"(kBlk), "h"(kThr), "r"(sfb), "h"(kBlk), "h"(kThr));
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) { s += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (s == 1234.5f) { sink[0] = s; }
}

template <class F>
void run(const char* name, F kernel, double k, float* sink) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    kernel<<<BLOCKS, WARPS * 32>>>(sink);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 4; ++r) { kernel<<<BLOCKS, WARPS * 32>>>(sink); }
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    // Один mma m16n8kK на варп = 2*16*8*K операций.
    const double ops = 2.0 * 16.0 * 8.0 * k * CHAINS * ITERS * WARPS * BLOCKS * 4.0;
    printf("%-6s %8.1f TOP/s  (%.3f мс на 4 запуска)\n", name, ops / (ms * 1e-3) / 1e12, ms);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  ошибка: %s\n", cudaGetErrorString(e)); }
}

int main() {
    float* sink = nullptr;
    cudaMalloc(&sink, sizeof(float));
    run("bf16", bf16_peak, 16.0, sink);
    run("fp8", fp8_peak, 32.0, sink);
    run("s8", s8_peak, 32.0, sink);
    run("nvfp4", nvfp4_peak, 64.0, sink);
    return 0;
}
