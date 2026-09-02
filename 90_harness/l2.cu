#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    int mclk = 0, bus = 0;
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("L2 = %.1f MB, SM = %d, shared/SM = %zu KB, regs/SM = %d\n",
           p.l2CacheSize / 1048576.0, p.multiProcessorCount,
           p.sharedMemPerMultiprocessor / 1024, p.regsPerMultiprocessor);
    printf("bus = %d bit, memClock = %d kHz -> teoreticheskiy pik = %.1f GB/s\n",
           bus, mclk, 2.0 * mclk * 1000.0 * (bus / 8) / 1e9);
    return 0;
}
