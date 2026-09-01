// Потолок с кормлением в настоящей пропорции ядра: на каждый шаг по K грузятся 4 фрагмента A
// (ldmatrix.x4) и 8 фрагментов B (ldmatrix.x2), и они кормят 4x8 = 32 MMA. Ровно та раскладка,
// что у nvfp4_w4a4_tma_kernel и у fp8_a8_tma_kernel: kMmaM=4, kMmaN=8.
#include <cstdio>
#include <cuda_runtime.h>

#define ITERS 512
#define MM 4
#define NN 8
#define WARPS 8
#define BLOCKS 170
#define SMEM_ROWS 128
#define ROW_BYTES 64

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
#define LDM4(d0, d1, d2, d3, addr)                                                                 \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"                 \
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                          \
                 : "r"(addr))
#define LDM2(d0, d1, addr)                                                                         \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"                       \
                 : "=r"(d0), "=r"(d1)                                                              \
                 : "r"(addr))

__global__ __launch_bounds__(WARPS * 32, 1) void fp8_tile(float* sink) {
    __shared__ unsigned char buf[SMEM_ROWS * ROW_BYTES];
    for (int i = threadIdx.x; i < SMEM_ROWS * ROW_BYTES / 4; i += blockDim.x) {
        reinterpret_cast<unsigned*>(buf)[i] = 0x3c3c3c3cu ^ i;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    float c[MM][NN][4];
#pragma unroll
    for (int m = 0; m < MM; ++m)
#pragma unroll
        for (int n = 0; n < NN; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.0f;
    for (int it = 0; it < ITERS; ++it) {
        unsigned af[MM][4], bf[NN][2];
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            LDM4(af[m][0], af[m][1], af[m][2], af[m][3],
                 smem_u32(&buf[((lane + m * 16 + it) & (SMEM_ROWS - 1)) * ROW_BYTES]));
        }
#pragma unroll
        for (int n = 0; n < NN; ++n) {
            LDM2(bf[n][0], bf[n][1],
                 smem_u32(&buf[((lane + n * 8 + it + 33) & (SMEM_ROWS - 1)) * ROW_BYTES]));
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
#pragma unroll
            for (int n = 0; n < NN; ++n) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[m][n][0]), "+f"(c[m][n][1]), "+f"(c[m][n][2]),
                               "+f"(c[m][n][3])
                             : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]),
                               "r"(bf[n][0]), "r"(bf[n][1]));
            }
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int m = 0; m < MM; ++m)
#pragma unroll
        for (int n = 0; n < NN; ++n) s += c[m][n][0] + c[m][n][1] + c[m][n][2] + c[m][n][3];
    if (s == 1234.5f) { sink[0] = s; }
}

__global__ __launch_bounds__(WARPS * 32, 1) void nvfp4_tile(float* sink) {
    __shared__ unsigned char buf[SMEM_ROWS * ROW_BYTES];
    for (int i = threadIdx.x; i < SMEM_ROWS * ROW_BYTES / 4; i += blockDim.x) {
        reinterpret_cast<unsigned*>(buf)[i] = 0x3c3c3c3cu ^ i;
    }
    __syncthreads();
    const int lane                = threadIdx.x & 31;
    constexpr unsigned short kBlk = 0;
    constexpr unsigned short kThr = 0;
    float c[MM][NN][4];
#pragma unroll
    for (int m = 0; m < MM; ++m)
#pragma unroll
        for (int n = 0; n < NN; ++n) c[m][n][0] = c[m][n][1] = c[m][n][2] = c[m][n][3] = 0.0f;
    for (int it = 0; it < ITERS; ++it) {
        unsigned af[MM][4], bf[NN][2], sa[MM], sb[NN];
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            LDM4(af[m][0], af[m][1], af[m][2], af[m][3],
                 smem_u32(&buf[((lane + m * 16 + it) & (SMEM_ROWS - 1)) * ROW_BYTES]));
            sa[m] = reinterpret_cast<unsigned*>(buf)[(lane + m + it) & 255];
        }
#pragma unroll
        for (int n = 0; n < NN; ++n) {
            LDM2(bf[n][0], bf[n][1],
                 smem_u32(&buf[((lane + n * 8 + it + 33) & (SMEM_ROWS - 1)) * ROW_BYTES]));
            sb[n] = reinterpret_cast<unsigned*>(buf)[(lane + n + it + 7) & 255];
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
#pragma unroll
            for (int n = 0; n < NN; ++n) {
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                             "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                             "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                             : "+f"(c[m][n][0]), "+f"(c[m][n][1]), "+f"(c[m][n][2]),
                               "+f"(c[m][n][3])
                             : "r"(af[m][0]), "r"(af[m][1]), "r"(af[m][2]), "r"(af[m][3]),
                               "r"(bf[n][0]), "r"(bf[n][1]), "r"(sa[m]), "h"(kBlk), "h"(kThr),
                               "r"(sb[n]), "h"(kBlk), "h"(kThr));
            }
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int m = 0; m < MM; ++m)
#pragma unroll
        for (int n = 0; n < NN; ++n) s += c[m][n][0] + c[m][n][1] + c[m][n][2] + c[m][n][3];
    if (s == 1234.5f) { sink[0] = s; }
}

template <class F> void run(const char* name, F kernel, double k, float* sink) {
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
    const double ops = 2.0 * 16.0 * 8.0 * k * MM * NN * ITERS * WARPS * BLOCKS * 4.0;
    printf("%-12s %8.1f TOP/s  (%.3f мс)\n", name, ops / (ms * 1e-3) / 1e12, ms);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  ошибка: %s\n", cudaGetErrorString(e)); }
}

int main() {
    float* sink = nullptr;
    cudaMalloc(&sink, sizeof(float));
    run("fp8 4x8 occ1", fp8_tile, 32.0, sink);
    run("nvfp4 4x8 occ1", nvfp4_tile, 64.0, sink);
    return 0;
}
