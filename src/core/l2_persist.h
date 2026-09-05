#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace ninfer::l2p {

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
 * Does nothing - no window and no reserve - when the range is wider than
 * cudaDeviceProp::accessPolicyMaxWindowSize, because past that limit the window covers only a
 * prefix of the range while the reserve is still taken in full. Returns whether a window was
 * installed, so the caller knows whether it has anything to take off the stream.
 */
bool pin_range(cudaStream_t stream, const void* base, std::size_t bytes);

/** Removes the window from the stream, so eager phases run under the default policy. */
void unpin(cudaStream_t stream);

} // namespace ninfer::l2p
