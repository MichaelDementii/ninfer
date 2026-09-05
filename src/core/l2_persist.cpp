#include "core/l2_persist.h"

#include <atomic>
#include <cstdio>

namespace ninfer::l2p {

bool pin_range(cudaStream_t stream, const void* base, std::size_t bytes) {
    if (base == nullptr || bytes == 0) { return false; }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return false; }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { return false; }
    if (prop.persistingL2CacheMaxSize <= 0 || prop.accessPolicyMaxWindowSize <= 0) { return false; }

    // An access policy window can never be longer than accessPolicyMaxWindowSize. A pool wider
    // than that does not get a shorter window over the whole pool, it gets a window over the
    // pool's prefix, while the device-wide reserve is taken in full and paid for everywhere,
    // prefill included. On a target where that happens the reservation is a pure loss, so do
    // nothing at all: no window, and no reserve either.
    if (bytes > static_cast<std::size_t>(prop.accessPolicyMaxWindowSize)) {
        // Nothing else reports this: the driver returns cudaSuccess whether the window covers
        // the pool or a fraction of it, so a silently truncated window is indistinguishable
        // from a working one. Say it once per process instead.
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true)) {
            std::fprintf(stderr,
                         "ninfer: L2 persistence disabled: the %zu byte Linear Attention state "
                         "pool is wider than the %zu byte access policy window of this device\n",
                         bytes, static_cast<std::size_t>(prop.accessPolicyMaxWindowSize));
        }
        return false;
    }

    const std::size_t window = bytes;
    std::size_t reserve      = window;
    if (reserve > static_cast<std::size_t>(prop.persistingL2CacheMaxSize)) {
        reserve = static_cast<std::size_t>(prop.persistingL2CacheMaxSize);
    }
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve) != cudaSuccess) {
        return false;
    }

    float hit_ratio = 1.0F;
    if (window > reserve) {
        hit_ratio = static_cast<float>(static_cast<double>(reserve) / static_cast<double>(window));
    }

    cudaStreamAttrValue value{};
    value.accessPolicyWindow.base_ptr  = const_cast<void*>(base);
    value.accessPolicyWindow.num_bytes = window;
    value.accessPolicyWindow.hitRatio  = hit_ratio;
    value.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    value.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
    (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &value);
    return true;
}

void unpin(cudaStream_t stream) {
    cudaStreamAttrValue value{};
    value.accessPolicyWindow.base_ptr  = nullptr;
    value.accessPolicyWindow.num_bytes = 0;
    value.accessPolicyWindow.hitRatio  = 0.0F;
    value.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    value.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
    (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &value);
}

} // namespace ninfer::l2p
