// Распаковка nvfp4: что генерит текущий static_cast<float2>(__nv_fp4x2_e2m1) и что даёт
// одна аппаратная инструкция cvt.rn.f16x2.e2m1x2. Проверяются и синтаксис, и совпадение
// значений на всех 256 байтах.
#include <cstdio>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float2 decode_current(unsigned char storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float2 decode_ptx(unsigned char storage) {
    unsigned packed;
    asm("cvt.rn.f16x2.e2m1x2 %0, %1;" : "=r"(packed) : "h"((unsigned short)storage));
    const __half2 h = *reinterpret_cast<const __half2*>(&packed);
    return __half22float2(h);
}

__global__ void check(float* out) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 256) { return; }
    const float2 a = decode_current((unsigned char)i);
    const float2 b = decode_ptx((unsigned char)i);
    out[i * 4 + 0] = a.x;
    out[i * 4 + 1] = a.y;
    out[i * 4 + 2] = b.x;
    out[i * 4 + 3] = b.y;
}

int main() {
    float* d;
    cudaMalloc(&d, 256 * 4 * sizeof(float));
    check<<<1, 256>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("cuda: %s\n", cudaGetErrorString(e));
        return 1;
    }
    float h[256 * 4];
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 256; ++i) {
        const bool same = h[i * 4 + 0] == h[i * 4 + 2] && h[i * 4 + 1] == h[i * 4 + 3];
        if (!same) {
            if (bad < 6) {
                printf("bayt %3d: tekushchiy (%g, %g)  ptx (%g, %g)\n", i, h[i * 4 + 0],
                       h[i * 4 + 1], h[i * 4 + 2], h[i * 4 + 3]);
            }
            ++bad;
        }
    }
    printf(bad == 0 ? "vse 256 baytov sovpali pobitovo\n" : "RASHOZHDENIY: %d\n", bad);
    return 0;
}
