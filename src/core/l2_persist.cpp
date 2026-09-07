#include "core/l2_persist.h"

#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>

namespace ninfer::l2p {
namespace {

// The limit is one number per device, so its owners have to be reconciled in one place. Requests
// are counted rather than replaced: withdrawing one owner must not shrink the set-aside another
// owner is still replaying a graph under.
struct Registry {
    std::mutex mutex;
    std::map<int, std::multiset<std::size_t>> live;
    // The limit in force before the first request on that device, restored when the last one goes
    // away. Recorded under the same lock that publishes the first request, so a second owner
    // cannot record an already raised limit as the one to return to, and kept until a restore
    // actually lands: a failed write must not lose the value it was going to write.
    std::map<int, std::size_t> baseline;
};

// Deliberately never destroyed. A handle owned by an object with static storage duration would
// otherwise reach a destroyed registry during exit, and the cost of the leak is one map.
Registry& registry() {
    static Registry* const instance = new Registry();
    return *instance;
}

// cudaDeviceSetLimit writes the calling thread's current device, so a release has to name the
// device its request was made against rather than whichever one happens to be bound. A failed
// bind is reported rather than swallowed: writing the limit anyway would apply one device's
// bookkeeping to another.
class BoundDevice {
public:
    explicit BoundDevice(int device) {
        int current = 0;
        if (cudaGetDevice(&current) != cudaSuccess) { return; }
        if (current == device) {
            bound_ = true;
            return;
        }
        if (cudaSetDevice(device) != cudaSuccess) { return; }
        previous_ = current;
        bound_    = true;
    }

    ~BoundDevice() {
        if (previous_ >= 0) { (void)cudaSetDevice(previous_); }
    }

    [[nodiscard]] bool bound() const noexcept { return bound_; }

    BoundDevice(const BoundDevice&)            = delete;
    BoundDevice& operator=(const BoundDevice&) = delete;
    BoundDevice(BoundDevice&&)                 = delete;
    BoundDevice& operator=(BoundDevice&&)      = delete;

private:
    int previous_ = -1;
    bool bound_   = false;
};

bool apply_limit(int device, std::size_t bytes) {
    const BoundDevice bound(device);
    if (!bound.bound()) { return false; }
    return cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes) == cudaSuccess;
}

// Drops one request and writes back what that device should hold now. Called with the lock held.
// When that was the last request, the baseline is given up only once the restoring write lands, so
// a device whose limit could not be written keeps the value it still owes a restore to. A device
// that still has owners needs no such care: the next withdrawal recomputes the target from what
// remains, so a write that did not land is corrected rather than remembered.
void withdraw(Registry& reg, int device, std::size_t bytes) {
    const auto slot = reg.live.find(device);
    if (slot == reg.live.end()) { return; }
    auto& live    = slot->second;
    const auto it = live.find(bytes);
    if (it == live.end()) { return; }
    live.erase(it);

    const auto base = reg.baseline.find(device);
    if (!live.empty()) {
        (void)apply_limit(device, *live.rbegin());
        return;
    }
    if (base == reg.baseline.end()) {
        // No recorded baseline means nothing is known to restore to; leaving the limit alone is
        // the only honest option. Unreachable while a request was live, kept as a guard.
        reg.live.erase(slot);
        return;
    }
    if (apply_limit(device, base->second)) {
        reg.live.erase(slot);
        reg.baseline.erase(base);
    }
}

} // namespace

Reservation::Reservation(Reservation&& other) noexcept {
    const std::lock_guard<std::mutex> guard(registry().mutex);
    bytes_       = other.bytes_;
    device_      = other.device_;
    held_        = other.held_;
    other.bytes_ = 0;
    other.held_  = false;
}

Reservation& Reservation::operator=(Reservation&& other) noexcept {
    if (this == &other) { return *this; }
    Registry& reg = registry();
    const std::lock_guard<std::mutex> guard(reg.mutex);
    if (held_) { withdraw(reg, device_, bytes_); }
    bytes_       = other.bytes_;
    device_      = other.device_;
    held_        = other.held_;
    other.bytes_ = 0;
    other.held_  = false;
    return *this;
}

bool Reservation::request(std::size_t bytes) {
    // Zero is not a release: a handle that asks for nothing still owes its graph the set-aside it
    // already holds.
    if (bytes == 0) { return false; }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return false; }

    Registry& reg = registry();
    const std::lock_guard<std::mutex> guard(reg.mutex);
    if (held_ && device_ == device && bytes_ == bytes) { return true; }

    auto& live = reg.live[device];
    // A withdrawal whose write did not land leaves an empty request set behind but keeps the
    // baseline it still owes a restore to, so an empty set is not proof that this call is the
    // first owner. Only a call that records the baseline may take it back on failure.
    const bool recorded_baseline = reg.baseline.find(device) == reg.baseline.end();
    if (recorded_baseline) {
        std::size_t current = 0;
        const BoundDevice bound(device);
        if (!bound.bound() ||
            cudaDeviceGetLimit(&current, cudaLimitPersistingL2CacheSize) != cudaSuccess) {
            if (live.empty()) { reg.live.erase(device); }
            return false;
        }
        reg.baseline[device] = current;
    }

    // Put the new request in force before withdrawing the old one. A refusal must not leave a
    // graph that is already captured running under a set-aside that has been handed back.
    live.insert(bytes);
    if (!apply_limit(device, *live.rbegin())) {
        const auto it = live.find(bytes);
        if (it != live.end()) { live.erase(it); }
        if (live.empty()) {
            reg.live.erase(device);
            if (recorded_baseline) { reg.baseline.erase(device); }
        }
        return false;
    }

    if (held_) { withdraw(reg, device_, bytes_); }
    bytes_  = bytes;
    device_ = device;
    held_   = true;
    return true;
}

void Reservation::release() noexcept {
    Registry& reg = registry();
    const std::lock_guard<std::mutex> guard(reg.mutex);
    if (!held_) { return; }
    held_ = false;
    withdraw(reg, device_, bytes_);
    bytes_ = 0;
}

bool pin_range(cudaStream_t stream, const void* base, std::size_t bytes, Reservation& reservation) {
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
    if (!reservation.request(reserve)) { return false; }

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
