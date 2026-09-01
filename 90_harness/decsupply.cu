// Сколько из времени декодного nvfp4-ядра занимает один подвоз весов.
// Сетка, размер блока, раскладка чтений и число фаз — как у nvfp4_linear_swiglu_small_t
// при T=4: 2176 CTA по 256 нитей, восемь варпов, каждый варп ведёт две строки по 5120
// значений, десять фаз по 16 значений на нить. Счёта нет, активаций нет.
//   VARIANT 0 — только коды и масштабы, как в ядре
//   VARIANT 1 — то же плюс барьер на каждой фазе, как у ветки SharedPhase
//   VARIANT 2 — то же плюс стейджинг активаций в общую память и барьер (полная ветка)
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define INTERMEDIATE 17408
#define K_ELEMS 5120
#define WARPS 8
#define THREADS (WARPS * 32)
#define BLOCKS (INTERMEDIATE / WARPS)
#define VALUES_PER_LANE 16
#define VALUES_PER_PHASE (32 * VALUES_PER_LANE)
#define PHASES (K_ELEMS / VALUES_PER_PHASE)
#define CODE_BYTES_PER_ROW (K_ELEMS / 2)
#define GROUPS_PER_ROW (K_ELEMS / 16)
#define TOKENS 4
#define COPIES 8

struct Pack {
    unsigned words[2];
};

template <int VARIANT>
__global__ __launch_bounds__(THREADS, 1) void supply(const unsigned char* __restrict__ codes_base,
                                                     const unsigned char* __restrict__ scales_base,
                                                     int copy,
                                                     const __nv_bfloat16* __restrict__ x,
                                                     float* __restrict__ sink) {
    __shared__ __align__(16) unsigned char activation[TOKENS * VALUES_PER_PHASE * 2];

    const unsigned char* codes =
        codes_base + (long long)copy * 2 * INTERMEDIATE * CODE_BYTES_PER_ROW;
    const unsigned char* scales = scales_base + (long long)copy * 2 * INTERMEDIATE * GROUPS_PER_ROW;
    const int lane      = threadIdx.x & 31;
    const int warp      = threadIdx.x >> 5;
    const int gate_row  = blockIdx.x * WARPS + warp;
    const int rows[2]   = {gate_row, gate_row + INTERMEDIATE};
    float accumulator   = 0.0f;

#pragma unroll 1
    for (int phase = 0; phase < PHASES; ++phase) {
        if constexpr (VARIANT == 2) {
            constexpr int kPacksPerToken = VALUES_PER_PHASE / 8;
            constexpr int kStagePacks    = TOKENS * kPacksPerToken;
            auto* destination            = reinterpret_cast<uint4*>(activation);
            for (int task = threadIdx.x; task < kStagePacks; task += THREADS) {
                const int local_token = task / kPacksPerToken;
                const int local_pack  = task - local_token * kPacksPerToken;
                destination[task]     = *reinterpret_cast<const uint4*>(
                    x + (long long)local_token * K_ELEMS + phase * VALUES_PER_PHASE +
                    local_pack * 8);
            }
        }
        if constexpr (VARIANT >= 1) { __syncthreads(); }

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const long long code_offset = (long long)rows[r] * CODE_BYTES_PER_ROW +
                                          phase * (VALUES_PER_PHASE / 2) + lane * (VALUES_PER_LANE / 2);
            const Pack pack = *reinterpret_cast<const Pack*>(codes + code_offset);
            const long long scale_offset =
                (long long)rows[r] * GROUPS_PER_ROW + phase * (VALUES_PER_PHASE / 16) + lane;
            const unsigned char coefficient = scales[scale_offset];
            accumulator += (float)(pack.words[0] ^ pack.words[1]) + (float)coefficient;
        }
    }
    // Сток, который компилятор не может доказать недостижимым, иначе он выбрасывает все чтения.
    if (accumulator != accumulator) { sink[0] = accumulator; }
}

template <int VARIANT, class K> static double run(const char* name, K kernel,
                                                  const unsigned char* c, const unsigned char* s,
                                                  const __nv_bfloat16* x, float* sink,
                                                  double base) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    kernel<<<BLOCKS, THREADS>>>(c, s, 0, x, sink);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < COPIES; ++r) { kernel<<<BLOCKS, THREADS>>>(c, s, r, x, sink); }
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us    = ms * 1000.0 / COPIES;
    const double bytes = (double)BLOCKS * WARPS * 2 * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    printf("%-46s %8.2f mks  %7.1f GB/s", name, us, bytes / (us * 1e-6) / 1e9);
    if (base > 0.0) { printf("   %5.1f%% ot yadra", 100.0 * us / base); }
    printf("\n");
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  cuda: %s\n", cudaGetErrorString(e)); }
    return us;
}

int main() {
    unsigned char *codes, *scales;
    __nv_bfloat16* x;
    float* sink;
    cudaMalloc(&codes, (size_t)COPIES * 2 * INTERMEDIATE * CODE_BYTES_PER_ROW);
    cudaMalloc(&scales, (size_t)COPIES * 2 * INTERMEDIATE * GROUPS_PER_ROW);
    cudaMalloc(&x, (size_t)TOKENS * K_ELEMS * sizeof(__nv_bfloat16));
    cudaMalloc(&sink, sizeof(float));
    cudaMemset(codes, 0x22, (size_t)COPIES * 2 * INTERMEDIATE * CODE_BYTES_PER_ROW);
    cudaMemset(scales, 0x33, (size_t)COPIES * 2 * INTERMEDIATE * GROUPS_PER_ROW);
    cudaMemset(x, 0x11, (size_t)TOKENS * K_ELEMS * sizeof(__nv_bfloat16));

    const double bytes = (double)BLOCKS * WARPS * 2 * (CODE_BYTES_PER_ROW + GROUPS_PER_ROW);
    printf("dekodnyy nvfp4: setka %d CTA x %d nitey, %d faz, %.1f MB vesov na pusk\n", BLOCKS,
           THREADS, PHASES, bytes / 1e6);
    printf("nastoyashchee yadro pri T=4 idet 81.92 mks (1226 GB/s, 68%% pika)\n\n");

    const double kernel_us = 81.92;
    run<0>("tolko kody i masshtaby", supply<0>, codes, scales, x, sink, kernel_us);
    run<1>("+ barer na kazhdoy faze", supply<1>, codes, scales, x, sink, kernel_us);
    run<2>("+ steyding aktivaciy i barer", supply<2>, codes, scales, x, sink, kernel_us);
    printf("\npovtor:\n");
    run<0>("tolko kody i masshtaby", supply<0>, codes, scales, x, sink, kernel_us);
    run<2>("+ steyding aktivaciy i barer", supply<2>, codes, scales, x, sink, kernel_us);
    return 0;
}
