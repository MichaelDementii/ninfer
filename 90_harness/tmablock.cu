// Что именно тормозит подвоз в раскладке nvfp4: коды или масштабы. Четыре варианта переносят
// одну и ту же стадию по частям. Дескриптор масштабов активаций описывает коробку 16 байт при шаге
// строки 1088 — это 256 крошечных запросов на стадию; у кодов коробка 64 байта.
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

// WHAT: 1 = только коды, 2 = коды + масштабы весов, 3 = коды + масштабы активаций, 4 = всё
template <int WHAT>
__global__ __launch_bounds__(THREADS, 1) void supply(const __grid_constant__ Maps maps,
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
    constexpr unsigned kBytes =
        BM * 64 + BN * 64 + ((WHAT == 3 || WHAT == 4 || WHAT == 5) ? BM * 16 : 0) +
        ((WHAT == 2 || WHAT == 4) ? BN * 8 : 0);

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
                tma2d(b_codes + s * BN * 64, &maps.b_codes, kt * 64, row_begin, &full[s]);
                if constexpr (WHAT == 3 || WHAT == 4) {
                    tma2d(a_scale4 + s * BM * 16, &maps.a_scales, (kt / 2) * 16, token_begin,
                          &full[s]);
                }
                if constexpr (WHAT == 5) {
                    const int tile = (token_begin / BM) * (KTILES / 2) + kt / 2;
                    tma2d(a_scale4 + s * BM * 16, &maps.a_scales, 0, tile * 16, &full[s]);
                }
                if constexpr (WHAT == 2 || WHAT == 4) {
                    tma2d(b_scale + s * BN * 8, &maps.b_scales, 0,
                          ((row_begin / 128) * 136 + kt * 2) * 32, &full[s]);
                }
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

static CUtensorMap make(void* p, cuuint64_t cols, cuuint64_t rows, cuuint64_t stride,
                        cuuint32_t bc, cuuint32_t br, CUtensorMapSwizzle sw) {
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

template <int WHAT, class K> static void run(const char* name, K kernel, Maps maps, float* sink) {
    const int smem = STAGES * (BM * 64 + BN * 64 + BM * 16 + BN * 8) + 2 * STAGES * 8 + 128;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(N_ROWS / BN, TOKENS / BM);
    const unsigned bytes_stage =
        BM * 64 + BN * 64 + ((WHAT == 3 || WHAT == 4 || WHAT == 5) ? BM * 16 : 0) +
        ((WHAT == 2 || WHAT == 4) ? BN * 8 : 0);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    kernel<<<grid, THREADS, smem>>>(maps, sink);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < 4; ++r) { kernel<<<grid, THREADS, smem>>>(maps, sink); }
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    const double bytes = (double)grid.x * grid.y * KTILES * bytes_stage;
    printf("%-30s %7.1f мкс, %6.0f ГБ/с, стадия %u Б\n", name, ms * 1000.0 / 4.0,
           bytes * 4.0 / (ms * 1e-3) / 1e9, bytes_stage);
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

    Maps maps{};
    maps.a_codes =
        make(ac, CODE_ROW_BYTES, TOKENS, CODE_ROW_BYTES, 64, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.b_codes =
        make(bc, CODE_ROW_BYTES, N_ROWS, CODE_ROW_BYTES, 64, BN, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.a_scales =
        make(as, GROUPS_PER_ROW, TOKENS, GROUPS_PER_ROW, 16, BM, CU_TENSOR_MAP_SWIZZLE_NONE);
    maps.b_scales = make(bs, 16, (size_t)N_ROWS * GROUPS_PER_ROW / 16, 16, 16, 64,
                         CU_TENSOR_MAP_SWIZZLE_NONE);

    run<1>("только коды", supply<1>, maps, sink);
    run<2>("коды + масштабы весов", supply<2>, maps, sink);
    run<3>("коды + масштабы активаций", supply<3>, maps, sink);
    run<4>("всё, как в ядре", supply<4>, maps, sink);
    Maps blocked = maps;
    blocked.a_scales = make(as, 256, (cuuint64_t)(TOKENS / BM) * (KTILES / 2) * 16, 256, 256, 16,
                            CU_TENSOR_MAP_SWIZZLE_NONE);
    run<5>("коды + масштабы блоком", supply<5>, blocked, sink);
    return 0;
}
