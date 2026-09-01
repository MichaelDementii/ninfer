// Два оставшихся дефекта подвоза в nvfp4-ядре после пакета 12.
//   A. Масштабы активаций возят вдвое больше, чем потребляет стадия: тайл на 16 групп берётся
//      на каждом k-тайле, а K128 съедает только восемь из шестнадцати байт.
//   B. Масштабы весов лежат сплошными 1024 байтами, но коробка описана как 16 Б x 64 строки,
//      то есть 64 крошечных запроса вместо четырёх по 256 Б.
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

// AHALF: 0 = тайл на 16 групп на каждом k-тайле (как сейчас), 1 = тайл ровно на 8 групп
// WWIDE: 0 = коробка масштабов весов 16x64 (как сейчас), 1 = 256x4
template <int AHALF, int WWIDE>
__global__ __launch_bounds__(THREADS, 1) void supply(const __grid_constant__ Maps maps,
                                                     float* sink) {
    extern __shared__ __align__(128) unsigned char raw[];
    constexpr int kAScaleBytes = AHALF ? BM * 8 : BM * 16;
    unsigned char* a_codes     = raw;
    unsigned char* b_codes     = a_codes + STAGES * BM * 64;
    unsigned char* a_scale4    = b_codes + STAGES * BN * 64;
    unsigned char* b_scale     = a_scale4 + STAGES * kAScaleBytes;
    unsigned long long* full   = reinterpret_cast<unsigned long long*>(b_scale + STAGES * BN * 8);
    unsigned long long* empty  = full + STAGES;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], CONSUMER_WARPS);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int token_begin     = static_cast<int>(blockIdx.y) * BM;
    const int row_begin       = static_cast<int>(blockIdx.x) * BN;
    constexpr unsigned kBytes = BM * 64 + BN * 64 + kAScaleBytes + BN * 8;

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
                if constexpr (AHALF) {
                    const int tile = (token_begin / BM) * (GROUPS_PER_ROW / 8) + kt;
                    tma2d(a_scale4 + s * kAScaleBytes, &maps.a_scales, 0, tile * 8, &full[s]);
                } else {
                    const int tile = (token_begin / BM) * (GROUPS_PER_ROW / 16) + kt / 2;
                    tma2d(a_scale4 + s * kAScaleBytes, &maps.a_scales, 0, tile * 16, &full[s]);
                }
                const int b_row = ((row_begin / 128) * (GROUPS_PER_ROW / 8) + kt * 2) * 32;
                if constexpr (WWIDE) {
                    tma2d(b_scale + s * BN * 8, &maps.b_scales, 0, b_row / 16, &full[s]);
                } else {
                    tma2d(b_scale + s * BN * 8, &maps.b_scales, 0, b_row, &full[s]);
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

template <int AHALF, int WWIDE, class K>
static double run(const char* name, K kernel, Maps maps, float* sink, double base) {
    constexpr int kAScaleBytes = AHALF ? BM * 8 : BM * 16;
    const int smem = STAGES * (BM * 64 + BN * 64 + kAScaleBytes + BN * 8) + 2 * STAGES * 8 + 128;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(N_ROWS / BN, TOKENS / BM);
    const unsigned bytes_stage = BM * 64 + BN * 64 + kAScaleBytes + BN * 8;
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
    const double us    = ms * 1000.0 / 4.0;
    printf("%-44s %7.1f mks  %6.0f GB/s  stadia %5u B", name, us,
           bytes * 4.0 / (ms * 1e-3) / 1e9, bytes_stage);
    if (base > 0.0) { printf("   x%.4f", us / base); }
    printf("\n");
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  cuda: %s\n", cudaGetErrorString(e)); }
    return us;
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
        make(bc, CODE_ROW_BYTES, N_ROWS, CODE_ROW_BYTES, 64, BN, CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap a16 = make(as, BM, (cuuint64_t)TOKENS / BM * (GROUPS_PER_ROW / 16) * 16, BM, BM, 16,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap a8  = make(as, BM, (cuuint64_t)TOKENS / BM * (GROUPS_PER_ROW / 8) * 8, BM, BM, 8,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap w64 = make(bs, 16, wbytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap w4  = make(bs, 256, wbytes / 256, 256, 256, 4, CU_TENSOR_MAP_SWIZZLE_NONE);

    Maps m16{acm, bcm, a16, w64};
    Maps m16w{acm, bcm, a16, w4};
    Maps m8{acm, bcm, a8, w64};
    Maps m8w{acm, bcm, a8, w4};

    printf("nvfp4, stadia podvoza: dva ostavshihsya defekta. T=%d, K=%d, N=%d\n\n", TOKENS, K_ELEMS,
           N_ROWS);
    const double base = run<0, 0>("kak seychas (paket 12)", supply<0, 0>, m16, sink, 0.0);
    run<0, 1>("+ masshtaby vesov korobkoy 256x4", supply<0, 1>, m16w, sink, base);
    run<1, 0>("+ masshtaby aktivaciy taylom na 8 grupp", supply<1, 0>, m8, sink, base);
    run<1, 1>("oba", supply<1, 1>, m8w, sink, base);
    printf("\npovtor:\n");
    run<0, 0>("kak seychas (paket 12)", supply<0, 0>, m16, sink, base);
    run<1, 1>("oba", supply<1, 1>, m8w, sink, base);
    return 0;
}
