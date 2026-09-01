// Зависит ли скорость подвоза TMA от ширины коробки. Два варианта переносят ровно одинаковое
// число байт из одинаковых буферов в одинаковый по размеру shared; отличается только внутреннее
// измерение дескриптора: 64 байта со свизлом 64B против 128 байт со свизлом 128B, и вдвое меньшее
// число шагов у второго. Масштабы выброшены — меряется чистая скорость подвоза кодов.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define BM 256
#define BN 128
#define STAGES 2
#define CONSUMER_WARPS 8
#define PRODUCER_THREADS 128
#define THREADS (CONSUMER_WARPS * 32 + PRODUCER_THREADS)

#define K_ELEMS 17408
#define N_ROWS 5120
#define TOKENS 4096
#define CODE_ROW_BYTES (K_ELEMS / 2)

struct alignas(128) Maps2 {
    CUtensorMap a, b;
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

template <int BOX, int KTILES>
__global__ __launch_bounds__(THREADS, 1) void supply(const __grid_constant__ Maps2 maps,
                                                     float* sink) {
    extern __shared__ __align__(128) unsigned char raw[];
    unsigned char* a          = raw;
    unsigned char* b          = a + STAGES * BM * BOX;
    unsigned long long* full  = reinterpret_cast<unsigned long long*>(b + STAGES * BN * BOX);
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
                bar_expect(&full[s], (BM + BN) * BOX);
                tma2d(a + s * BM * BOX, &maps.a, kt * BOX, token_begin, &full[s]);
                tma2d(b + s * BN * BOX, &maps.b, kt * BOX, row_begin, &full[s]);
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

template <int BOX, int KTILES, class K>
static void run(const char* name, K kernel, void* ac, void* bc, float* sink, CUtensorMapSwizzle sw) {
    Maps2 maps{};
    maps.a = make(ac, CODE_ROW_BYTES, TOKENS, CODE_ROW_BYTES, BOX, BM, sw);
    maps.b = make(bc, CODE_ROW_BYTES, N_ROWS, CODE_ROW_BYTES, BOX, BN, sw);
    const int smem = STAGES * (BM + BN) * BOX + 2 * STAGES * 8 + 128;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(N_ROWS / BN, TOKENS / BM);
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
    const double bytes = (double)grid.x * grid.y * KTILES * (BM + BN) * BOX;
    printf("%-22s %7.1f мкс, %6.0f ГБ/с, shared %d Б, шагов %d\n", name, ms * 1000.0 / 4.0,
           bytes * 4.0 / (ms * 1e-3) / 1e9, smem, KTILES);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  cuda: %s\n", cudaGetErrorString(e)); }
}

int main() {
    void *ac, *bc;
    float* sink;
    cudaMalloc(&ac, (size_t)TOKENS * CODE_ROW_BYTES);
    cudaMalloc(&bc, (size_t)N_ROWS * CODE_ROW_BYTES);
    cudaMalloc(&sink, sizeof(float));
    cudaMemset(ac, 0x22, (size_t)TOKENS * CODE_ROW_BYTES);
    cudaMemset(bc, 0x22, (size_t)N_ROWS * CODE_ROW_BYTES);
    run<64, CODE_ROW_BYTES / 64>("коробка 64 Б, свизл 64", supply<64, CODE_ROW_BYTES / 64>, ac, bc,
                                 sink, CU_TENSOR_MAP_SWIZZLE_64B);
    run<128, CODE_ROW_BYTES / 128>("коробка 128 Б, свизл 128", supply<128, CODE_ROW_BYTES / 128>,
                                   ac, bc, sink, CU_TENSOR_MAP_SWIZZLE_128B);
    run<32, CODE_ROW_BYTES / 32>("коробка 32 Б, свизл 32", supply<32, CODE_ROW_BYTES / 32>, ac, bc,
                                 sink, CU_TENSOR_MAP_SWIZZLE_32B);
    return 0;
}
