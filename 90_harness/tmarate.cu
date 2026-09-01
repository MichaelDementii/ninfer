// Скорость подвоза TMA в раскладке nvfp4_w4a4_tma_kernel: те же четыре дескриптора, та же стадия
// 29696 Б, то же кольцо из трёх mbarrier, тот же грид. Консьюмеры только ждут и отмечаются —
// счёта нет. Если время близко к настоящему ядру, ядро упирается в подвоз.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define BM 256
#define BN 128
#define STAGES 3
#define CONSUMER_WARPS 8
#define PRODUCER_THREADS 128
#define THREADS (CONSUMER_WARPS * 32 + PRODUCER_THREADS)

#define K_ELEMS 17408
#define N_ROWS 5120
#define TOKENS 4096
#define CODE_ROW_BYTES (K_ELEMS / 2)
#define GROUPS_PER_ROW (K_ELEMS / 16)
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

__global__ __launch_bounds__(THREADS, 1) void producer_only(const __grid_constant__ Maps maps,
                                                            float* sink) {
    extern __shared__ __align__(128) unsigned char raw[];
    unsigned char* a_codes  = raw;
    unsigned char* b_codes  = a_codes + STAGES * BM * 64;
    unsigned char* a_scale4 = b_codes + STAGES * BN * 64;
    unsigned char* b_scale  = a_scale4 + STAGES * BM * 16;
    unsigned long long* full  = reinterpret_cast<unsigned long long*>(b_scale + STAGES * BN * 8);
    unsigned long long* empty = full + STAGES;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], CONSUMER_WARPS);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int token_begin = static_cast<int>(blockIdx.y) * BM;
    const int row_begin   = static_cast<int>(blockIdx.x) * BN;

    if (threadIdx.x < PRODUCER_THREADS) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x == 0) {
#pragma unroll 1
            for (int kt = 0; kt < KTILES; ++kt) {
                const int s          = kt % STAGES;
                const unsigned phase = 1u ^ ((kt / STAGES) & 1u);
                bar_wait(&empty[s], phase);
                bar_expect(&full[s], BM * 64 + BN * 64 + BM * 16 + BN * 2 * 4);
                tma2d(a_codes + s * BM * 64, &maps.a_codes, kt * 64, token_begin, &full[s]);
                tma2d(b_codes + s * BN * 64, &maps.b_codes, kt * 64, row_begin, &full[s]);
                tma2d(a_scale4 + s * BM * 16, &maps.a_scales, (kt / 2) * 16, token_begin, &full[s]);
                tma2d(b_scale + s * BN * 8, &maps.b_scales, 0,
                      ((row_begin / 128) * 136 + kt * 2) * 32, &full[s]);
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
        printf("ошибка %s: %s\n", what, n ? n : "?");
    }
}

static CUtensorMap make(void* p, unsigned long long cols, unsigned long long rows,
                        unsigned long long stride, unsigned bc, unsigned br,
                        CUtensorMapSwizzle sw) {
    CUtensorMap m{};
    unsigned long long gd[] = {cols, rows};
    unsigned long long gs[] = {stride};
    unsigned bd[]           = {bc, br};
    unsigned es[]           = {1, 1};
    check(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          "encode");
    return m;
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

    Maps maps{};
    maps.a_codes =
        make(ac, CODE_ROW_BYTES, TOKENS, CODE_ROW_BYTES, 64, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.b_codes =
        make(bc, CODE_ROW_BYTES, N_ROWS, CODE_ROW_BYTES, 64, BN, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.a_scales =
        make(as, GROUPS_PER_ROW, TOKENS, GROUPS_PER_ROW, 16, BM, CU_TENSOR_MAP_SWIZZLE_NONE);
    maps.b_scales = make(bs, 16, (size_t)N_ROWS * GROUPS_PER_ROW / 16, 16, 16, 64,
                         CU_TENSOR_MAP_SWIZZLE_NONE);

    const int smem = STAGES * (BM * 64 + BN * 64 + BM * 16 + BN * 8) + 2 * STAGES * 8 + 128;
    cudaFuncSetAttribute(producer_only, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(N_ROWS / BN, TOKENS / BM);

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    producer_only<<<grid, THREADS, smem>>>(maps, sink);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 4; ++r) { producer_only<<<grid, THREADS, smem>>>(maps, sink); }
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    const double bytes =
        (double)grid.x * grid.y * KTILES * (BM * 64 + BN * 64 + BM * 16 + BN * 8);
    printf("только подвоз: %.1f мкс на запуск, %.0f ГБ/с, shared %d Б\n", ms * 1000.0 / 4.0,
           bytes * 4.0 / (ms * 1e-3) / 1e9, smem);
    printf("для сравнения: настоящее ядро на этой форме — 712.7 мкс\n");
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("cuda: %s\n", cudaGetErrorString(e)); }
    return 0;
}
