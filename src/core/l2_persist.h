#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace ninfer::l2p {

/**
 * Ownership handle for a device's persisting-L2 set-aside.
 *
 * cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize) is context state, not stream or graph state:
 * it outlives both the access policy window that motivated it and the graph the window was baked
 * into, so it needs an owner that gives it back. A handle registers one request against one
 * device; the limit is held at the largest live request for that device, and returns to the limit
 * that was in force before the first request once the last one is released. Without that, work
 * started after the requesting owner is gone carries a set-aside while installing no window, which
 * costs what an unused reserve costs and buys nothing.
 *
 * The limit is per device, not per process, so requests are reconciled per device ordinal and a
 * release binds its own device before writing the limit back; a bind that fails is reported rather
 * than written to whichever device happens to be current. Handles are safe to use from several
 * threads.
 */
class Reservation {
public:
    Reservation() noexcept = default;

    ~Reservation() { release(); }

    Reservation(const Reservation&)            = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&& other) noexcept;
    Reservation& operator=(Reservation&& other) noexcept;

    /**
     * Requests `bytes` of set-aside on the calling thread's current device, replacing whatever
     * this handle asked for before. The limit becomes the largest request live on that device.
     *
     * Returns whether the request is in force. A refused request changes nothing: the previous
     * request of this handle stays live, because a graph may already be captured under it. A
     * request of zero bytes is refused for the same reason, rather than treated as a release.
     */
    bool request(std::size_t bytes);

    /** Withdraws this handle's request, lowering or restoring its device's limit accordingly. */
    void release() noexcept;

private:
    std::size_t bytes_ = 0;
    int device_        = 0;
    bool held_         = false;
};

/**
 * Pins a device range in L2 for the graph captured next on this stream.
 *
 * Must be called BEFORE cudaStreamBeginCapture: capture bakes the stream access policy window
 * into every kernel node of the graph and does not re-read the stream attribute at replay.
 * The reserve is clamped to cudaDeviceProp::persistingL2CacheMaxSize, and when the range is
 * larger than the reserve the hit ratio is lowered to the fraction that physically fits, which
 * is what CUDA requires: a window larger than the reserve at hitRatio 1 makes the lines evict
 * each other.
 *
 * The set-aside is taken through `reservation`, which must outlive every graph captured under this
 * window: the window is replayed from the graph long after this call returns, and the set-aside
 * has to still be in force for it to mean anything.
 *
 * Does nothing - no window and no reserve - when the range is wider than
 * cudaDeviceProp::accessPolicyMaxWindowSize, because past that limit the window covers only a
 * prefix of the range while the device-wide reserve is still taken in full. Returns whether a
 * window was installed, so the caller knows whether it has anything to take off the stream.
 */
bool pin_range(cudaStream_t stream, const void* base, std::size_t bytes, Reservation& reservation);

/** Removes the window from the stream, so eager phases run under the default policy. */
void unpin(cudaStream_t stream);

} // namespace ninfer::l2p
