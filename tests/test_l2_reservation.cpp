// The persisting-L2 set-aside is device-context state that outlives the access policy window and
// the graph the window is baked into. These cases pin the property that makes it safe to touch at
// all: whatever the handles do, the device is left with the limit it had before the first request.
#include "core/l2_persist.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {

int failures = 0;

std::size_t limit() {
    std::size_t value = 0;
    if (cudaDeviceGetLimit(&value, cudaLimitPersistingL2CacheSize) != cudaSuccess) {
        return static_cast<std::size_t>(-1);
    }
    return value;
}

void expect_at_least(const char* what, std::size_t got, std::size_t want) {
    if (got < want) {
        std::cout << what << ": limit " << got << " does not cover " << want << '\n';
        ++failures;
    }
}

// The set-aside has to come down as well as go up, and on a card whose driver default already
// exceeds a small request an "at least" check cannot see that. This is the direction that fails
// when a withdrawal stops lowering the limit to the request that survived it.
void expect_below(const char* what, std::size_t got, std::size_t want) {
    if (got >= want) {
        std::cout << what << ": limit " << got << " still carries " << want << '\n';
        ++failures;
    }
}

void expect_equal(const char* what, std::size_t got, std::size_t want) {
    if (got != want) {
        std::cout << what << ": limit " << got << ", expected " << want << '\n';
        ++failures;
    }
}

bool cuda_unavailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) != cudaSuccess || count <= 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.persistingL2CacheMaxSize <= 0) {
        std::cout << "SKIP: device reserves no L2 for persisting accesses\n";
        return 77;
    }

    // The driver rounds a set-aside up, so coverage is checked with >= and only the restored value
    // is checked for equality - it is the one the driver itself produced.
    const std::size_t ceiling = static_cast<std::size_t>(prop.persistingL2CacheMaxSize);
    const std::size_t small   = ceiling / 8;
    const std::size_t large   = ceiling / 2;
    if (small == 0) {
        std::cout << "SKIP: persisting L2 reserve too small to subdivide\n";
        return 77;
    }
    const std::size_t baseline = limit();

    {
        ninfer::l2p::Reservation outer;
        if (!outer.request(small)) {
            std::cout << "a request the device advertises room for was refused\n";
            return 1;
        }
        expect_at_least("one request covers what it asked for", limit(), small);
        {
            ninfer::l2p::Reservation inner;
            if (!inner.request(large)) {
                std::cout << "a nested request was refused\n";
                return 1;
            }
            expect_at_least("the larger of two live requests is in force", limit(), large);
        }
        expect_at_least("the survivor of two requests keeps its own", limit(), small);
        expect_below("the departed request is no longer in force", limit(), large);
    }
    expect_equal("the last release restores the limit it found", limit(), baseline);

    // Two handles asking for the same size are two requests, not one: withdrawing either must
    // leave the other covered. This is bookkeeping a set would get wrong and a multiset gets right.
    {
        ninfer::l2p::Reservation first;
        ninfer::l2p::Reservation second;
        if (!first.request(large) || !second.request(large)) {
            std::cout << "two handles of equal size were not both admitted\n";
            return 1;
        }
        first.release();
        expect_at_least("equal-sized twin still covered after its peer leaves", limit(), large);
    }
    expect_equal("equal-sized twins restore the limit once both leave", limit(), baseline);

    // Re-asking the same size must not ratchet the recorded baseline: it is read when the first
    // request goes in, and a cycle of release and request reads it again.
    {
        ninfer::l2p::Reservation handle;
        for (int cycle = 0; cycle < 32; ++cycle) {
            if (!handle.request(large)) {
                std::cout << "a repeated request was refused on cycle " << cycle << '\n';
                return 1;
            }
            expect_at_least("a repeated request stays in force", limit(), large);
            handle.release();
        }
    }
    expect_equal("32 request/release cycles do not drift the baseline", limit(), baseline);

    // Zero is not a release. A handle that is asked for nothing keeps what its graph is running
    // under, and says it did not take the request.
    {
        ninfer::l2p::Reservation handle;
        if (!handle.request(large)) {
            std::cout << "a request before the zero case was refused\n";
            return 1;
        }
        if (handle.request(0)) {
            std::cout << "a zero-byte request was reported as taken\n";
            ++failures;
        }
        expect_at_least("a zero-byte request leaves the live one alone", limit(), large);
    }
    expect_equal("the zero case restores the limit on release", limit(), baseline);

    {
        ninfer::l2p::Reservation source;
        if (!source.request(large)) {
            std::cout << "a request before the move case was refused\n";
            return 1;
        }
        ninfer::l2p::Reservation sink(std::move(source));
        expect_at_least("a moved handle carries its request", limit(), large);
        source.release(); // The moved-from handle owns nothing and must not disturb the limit.
        expect_at_least("releasing a moved-from handle changes nothing", limit(), large);
    }
    expect_equal("the moved-to handle restores the limit when it dies", limit(), baseline);

    // A live handle asked for a different size replaces its own request rather than adding one.
    // Nothing in the product exercises this today, but the header advertises it, and it is the
    // subtlest path in the file: the new request goes in before the old one comes out.
    {
        ninfer::l2p::Reservation handle;
        if (!handle.request(large) || !handle.request(small)) {
            std::cout << "a handle could not replace its own request\n";
            return 1;
        }
        expect_at_least("a replacement covers what it asked for", limit(), small);
        expect_below("the replaced request is not left standing", limit(), large);
    }
    expect_equal("the replacement case restores the limit on release", limit(), baseline);

    // A request the device cannot honour must be refused without disturbing a live one. This is
    // the only case that reaches the rollback in request(), because nothing else here makes the
    // write fail, and a rollback that gives back another owner's baseline is exactly the leak
    // this file exists to prevent.
    {
        ninfer::l2p::Reservation live_one;
        if (!live_one.request(large)) {
            std::cout << "a request before the refusal case was refused\n";
            return 1;
        }
        ninfer::l2p::Reservation refused;
        if (refused.request(ceiling * 2)) {
            std::cout << "a request beyond the device ceiling was reported as taken\n";
            ++failures;
        }
        expect_at_least("a refused request leaves the live one covered", limit(), large);
    }
    expect_equal("a refused request leaves nothing behind", limit(), baseline);

    // A refused request must not leave a baseline of its own behind. Nothing already here can see
    // that it did: the registry looks the same either way and the limit never moved. It becomes
    // visible only when the process already holds a set-aside of its own by the time the next
    // request arrives - then a baseline stashed by the refusal is handed back instead of that one.
    {
        ninfer::l2p::Reservation refused_cold;
        if (refused_cold.request(ceiling * 2)) {
            std::cout << "a cold request beyond the device ceiling was reported as taken\n";
            ++failures;
        }
    }
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, large) == cudaSuccess) {
        // Whatever the driver rounded that to is this process's own set-aside now, and it is what
        // a later release owes back - not the limit that was in force before the refusal.
        const std::size_t theirs = limit();
        {
            ninfer::l2p::Reservation ours;
            if (!ours.request(small)) {
                std::cout << "a request after the cold refusal was refused\n";
                return 1;
            }
        }
        expect_equal("a refused request does not stale the baseline", limit(), theirs);
        (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, baseline);
    }
    expect_equal("the cold refusal case leaves the limit as it found it", limit(), baseline);

    // The handle's own fields are part of what the registry lock protects. Before that was true,
    // two threads on one handle could publish two requests and withdraw one, stranding the device
    // at a raised limit with no window installed - the exact state this file exists to prevent.
    {
        ninfer::l2p::Reservation shared;
        std::vector<std::thread> threads;
        threads.reserve(4);
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&shared, large]() {
                for (int i = 0; i < 2000; ++i) {
                    (void)shared.request(large);
                    shared.release();
                }
            });
        }
        for (std::thread& thread : threads) { thread.join(); }
        shared.release();
    }
    expect_equal("concurrent use of one handle leaves no request behind", limit(), baseline);

    if (failures != 0) {
        std::cout << failures << " L2 reservation checks failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
