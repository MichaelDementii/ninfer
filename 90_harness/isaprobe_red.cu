// Проба ISA: какие формы red.global.add существуют на sm_120a.
// Приёмка — ptxas выходит с нулём и опкод виден в SASS. Кандидат 2 (слияние финализации top-8
// в эпилог down-GEMM) без векторной формы теряет большую часть смысла.
#include <cstdio>
#include <cuda_bf16.h>

__global__ void probe_f32(float* p, float v) {
    asm volatile("red.global.add.f32 [%0], %1;" : : "l"(p), "f"(v) : "memory");
}
__global__ void probe_v2f32(float* p, float a, float b) {
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" : : "l"(p), "f"(a), "f"(b) : "memory");
}
__global__ void probe_v4f32(float* p, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                 : : "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__global__ void probe_bf16x2(__nv_bfloat162* p, unsigned v) {
    asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" : : "l"(p), "r"(v) : "memory");
}
__global__ void probe_v2bf16x2(__nv_bfloat162* p, unsigned a, unsigned b) {
    asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};" : : "l"(p), "r"(a), "r"(b) : "memory");
}
__global__ void probe_release(float* p, float v) {
    asm volatile("red.release.gpu.global.add.f32 [%0], %1;" : : "l"(p), "f"(v) : "memory");
}

int main() {
    float* d = nullptr;
    cudaMalloc(&d, 64);
    cudaMemset(d, 0, 64);
    probe_f32<<<1, 32>>>(d, 1.0F);
    probe_v2f32<<<1, 32>>>(d, 1.0F, 2.0F);
    probe_v4f32<<<1, 32>>>(d, 1.0F, 2.0F, 3.0F, 4.0F);
    probe_bf16x2<<<1, 32>>>(reinterpret_cast<__nv_bfloat162*>(d), 0x3f803f80u);
    probe_v2bf16x2<<<1, 32>>>(reinterpret_cast<__nv_bfloat162*>(d), 0x3f803f80u, 0x3f803f80u);
    probe_release<<<1, 32>>>(d, 1.0F);
    cudaError_t e = cudaDeviceSynchronize();
    float h[4] = {};
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    std::printf("run=%s  h=[%g %g %g %g]\n", cudaGetErrorString(e), h[0], h[1], h[2], h[3]);
    return 0;
}
