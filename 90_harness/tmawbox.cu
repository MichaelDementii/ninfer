// Ширина коробки масштабов весов на форме слитого SwiGLU: два потока масштабов на стадию,
// разнесённых фиксированным большим шагом (ветви gate и up). На плоской геометрии поток один,
// и там широкая коробка выигрывает; здесь проверяется, что делает та же коробка при двух потоках.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define BM 256
#define BN 128
#define PAIRN 64
#define STAGES 3
#define CONSUMER_WARPS 8
#define PRODUCER_THREADS 128
#define THREADS (CONSUMER_WARPS * 32 + PRODUCER_THREADS)

#define K_ELEMS 5120
#define INTERMEDIATE 17408
#define N_ROWS (2 * INTERMEDIATE)
#define TOKENS 4096
#define CODE_ROW_BYTES (K_ELEMS / 2)
#define GROUPS_PER_ROW (K_ELEMS / 16)
#define SCALE_TILES_PER_ROW (GROUPS_PER_ROW / 8)
#define KTILES (K_ELEMS / 128)

struct alignas(128) Maps {
    CUtensorMap a_codes, b_codes, a_scales, b_scales;
};

__device__ __forceinline__ unsigned sm_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void bar_init(unsigned long long* b, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sm_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned long long* b, unsigned ph) {
    asm volatile("{\n.reg .pred d;\nL:\nmbarrier.try_wait.parity.shared::cta.b64 d, [%0], %1, "
                 "%2;\n@d bra E;\nbra L;\nE:\n}\n" ::"r"(sm_u32(b)),
                 "r"(ph), "r"(0x989680u)
                 : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sm_u32(b)) : "memory");
}
__device__ __forceinline__ void bar_expect(unsigned long long* b, unsigned n) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sm_u32(b)), "r"(n)
                 : "memory");
}
__device__ __forceinline__ void tma2d(void* d, const CUtensorMap* m, int c0, int c1,
                                      unsigned long long* b) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];" ::"r"(sm_u32(d)),
                 "l"(m), "r"(c0), "r"(c1), "r"(sm_u32(b))
                 : "memory");
}

// ROWB — сколько байт в строке дескриптора масштабов весов: 16 (сейчас), 64, 128 или 256
template <int ROWB>
__global__ __launch_bounds__(THREADS, 1) void supply(const __grid_constant__ Maps maps,
                                                     float* sink) {
    extern __shared__ __align__(128) unsigned char raw[];
    unsigned char* a_codes    = raw;
    unsigned char* b_codes    = a_codes + STAGES * BM * 64;
    unsigned char* a_scale4   = b_codes + STAGES * BN * 64;
    unsigned char* b_scale    = a_scale4 + STAGES * BM * 8;
    unsigned long long* full  = reinterpret_cast<unsigned long long*>(b_scale + STAGES * 2 * 1024);
    unsigned long long* empty = full + STAGES;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], CONSUMER_WARPS);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int token_begin     = static_cast<int>(blockIdx.y) * BM;
    const int pair_begin      = static_cast<int>(blockIdx.x) * PAIRN;
    constexpr unsigned kBytes = BM * 64 + BN * 64 + BM * 8 + 2 * 1024;

    if (threadIdx.x < PRODUCER_THREADS) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x == 0) {
#pragma unroll 1
            for (int kt = 0; kt < KTILES; ++kt) {
                const int s          = kt % STAGES;
                const unsigned phase = 1u ^ ((kt / STAGES) & 1u);
                bar_wait(&empty[s], phase);
                bar_expect(&full[s], kBytes);
                tma2d(a_codes + s * BM * 64, &maps.a_codes, kt * 64, token_begin, &full[s]);
                tma2d(b_codes + s * BN * 64, &maps.b_codes, kt * 64, pair_begin, &full[s]);
                tma2d(b_codes + s * BN * 64 + PAIRN * 64, &maps.b_codes, kt * 64,
                      pair_begin + INTERMEDIATE, &full[s]);
                const int tile = (token_begin / BM) * (GROUPS_PER_ROW / 8) + kt;
                tma2d(a_scale4 + s * BM * 8, &maps.a_scales, 0, tile * 8, &full[s]);
                const int gate = ((pair_begin / 128) * SCALE_TILES_PER_ROW + kt * 2) * 32 * 16;
                const int up   = (((pair_begin + INTERMEDIATE) / 128) * SCALE_TILES_PER_ROW +
                                kt * 2) *
                               32 * 16;
                tma2d(b_scale + s * 2048, &maps.b_scales, 0, gate / ROWB, &full[s]);
                tma2d(b_scale + s * 2048 + 1024, &maps.b_scales, 0, up / ROWB, &full[s]);
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    const int lane = (static_cast<int>(threadIdx.x) - PRODUCER_THREADS) & 31;
#pragma unroll 1
    for (int kt = 0; kt < KTILES; ++kt) {
        const int s          = kt % STAGES;
        const unsigned phase = (kt / STAGES) & 1u;
        bar_wait(&full[s], phase);
        if (lane == 0) { bar_arrive(&empty[s]); }
    }
    if (threadIdx.x == 99999u) { sink[0] = 1.0f; }
}

static void check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* n = nullptr;
        cuGetErrorName(r, &n);
        printf("oshibka %s: %s\n", what, n ? n : "?");
    }
}

static CUtensorMap make(void* p, cuuint64_t cols, cuuint64_t rows, cuuint64_t stride, cuuint32_t bc,
                        cuuint32_t br, CUtensorMapSwizzle sw) {
    CUtensorMap m{};
    cuuint64_t gd[] = {cols, rows};
    cuuint64_t gs[] = {stride};
    cuuint32_t bd[] = {bc, br};
    cuuint32_t es[] = {1, 1};
    check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "encode");
    return m;
}

template <int ROWB, class K> static void run(K kernel, Maps maps, float* sink, double base) {
    const int smem = STAGES * (BM * 64 + BN * 64 + BM * 8 + 2 * 1024) + 2 * STAGES * 8 + 128;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(INTERMEDIATE / PAIRN, TOKENS / BM);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    kernel<<<grid, THREADS, smem>>>(maps, sink);
    cudaDeviceSynchronize();
    double best = 1e30, worst = 0.0, total = 0.0;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(e0);
        kernel<<<grid, THREADS, smem>>>(maps, sink);
        cudaEventRecord(e1);
        cudaDeviceSynchronize();
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = ms * 1000.0;
        if (us < best) { best = us; }
        if (us > worst) { worst = us; }
        total += us;
    }
    printf("stroka %3d B, zaprosov na stadiyu %2d:  sred %7.1f  min %7.1f  max %7.1f mks", ROWB,
           1024 / ROWB, total / 5.0, best, worst);
    if (base > 0.0) { printf("   x%.4f", (total / 5.0) / base); }
    printf("\n");
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  cuda: %s\n", cudaGetErrorString(e)); }
}

int main() {
    void *ac, *bc, *as, *bs;
    float* sink;
    cudaMalloc(&ac, (size_t)TOKENS * CODE_ROW_BYTES);
    cudaMalloc(&bc, (size_t)N_ROWS * CODE_ROW_BYTES);
    cudaMalloc(&as, (size_t)TOKENS * GROUPS_PER_ROW);
    cudaMalloc(&bs, (size_t)N_ROWS * GROUPS_PER_ROW);
    cudaMalloc(&sink, sizeof(float));
    cudaMemset(ac, 0x22, (size_t)TOKENS * CODE_ROW_BYTES);
    cudaMemset(bc, 0x22, (size_t)N_ROWS * CODE_ROW_BYTES);
    cudaMemset(as, 0x33, (size_t)TOKENS * GROUPS_PER_ROW);
    cudaMemset(bs, 0x33, (size_t)N_ROWS * GROUPS_PER_ROW);

    const cuuint64_t wbytes = (cuuint64_t)N_ROWS * GROUPS_PER_ROW;
    CUtensorMap acm =
        make(ac, CODE_ROW_BYTES, TOKENS, CODE_ROW_BYTES, 64, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap bcm =
        make(bc, CODE_ROW_BYTES, N_ROWS, CODE_ROW_BYTES, 64, PAIRN, CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap a8 = make(as, BM, (cuuint64_t)TOKENS / BM * (GROUPS_PER_ROW / 8) * 8, BM, BM, 8,
                          CU_TENSOR_MAP_SWIZZLE_NONE);

    printf("forma slitogo SwiGLU: dva potoka masshtabov vesov na stadiyu\n\n");
    double base = 0.0;
    {
        CUtensorMap w = make(bs, 16, wbytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
        Maps m{acm, bcm, a8, w};
        const int smem = STAGES * (BM * 64 + BN * 64 + BM * 8 + 2 * 1024) + 2 * STAGES * 8 + 128;
        cudaFuncSetAttribute(supply<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        dim3 grid(INTERMEDIATE / PAIRN, TOKENS / BM);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        supply<16><<<grid, THREADS, smem>>>(m, sink);
        cudaDeviceSynchronize();
        double total = 0.0;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0);
            supply<16><<<grid, THREADS, smem>>>(m, sink);
            cudaEventRecord(e1);
            cudaDeviceSynchronize();
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            total += ms * 1000.0;
        }
        base = total / 5.0;
    }
    {
        CUtensorMap w = make(bs, 16, wbytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
        run<16>(supply<16>, Maps{acm, bcm, a8, w}, sink, base);
    }
    {
        CUtensorMap w = make(bs, 64, wbytes / 64, 64, 64, 16, CU_TENSOR_MAP_SWIZZLE_NONE);
        run<64>(supply<64>, Maps{acm, bcm, a8, w}, sink, base);
    }
    {
        CUtensorMap w = make(bs, 128, wbytes / 128, 128, 128, 8, CU_TENSOR_MAP_SWIZZLE_NONE);
        run<128>(supply<128>, Maps{acm, bcm, a8, w}, sink, base);
    }
    {
        CUtensorMap w = make(bs, 256, wbytes / 256, 256, 256, 4, CU_TENSOR_MAP_SWIZZLE_NONE);
        run<256>(supply<256>, Maps{acm, bcm, a8, w}, sink, base);
    }
    printf("\npovtor:\n");
    {
        CUtensorMap w = make(bs, 16, wbytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
        run<16>(supply<16>, Maps{acm, bcm, a8, w}, sink, base);
    }
    {
        CUtensorMap w = make(bs, 256, wbytes / 256, 256, 256, 4, CU_TENSOR_MAP_SWIZZLE_NONE);
        run<256>(supply<256>, Maps{acm, bcm, a8, w}, sink, base);
    }
    return 0;
}
