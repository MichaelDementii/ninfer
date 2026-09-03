#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_tma.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ninfer::ops::detail {

template <class Geometry>
struct Fp8LinearA8ProductionSchedule;

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

// TMA route. One CTA per SM with a 256-token tile. Four stages put the tensor pipeline at
// 4 * (256 + 128) * 64 = 96 KiB, which is the largest depth that fits under the 99 KiB per-CTA
// cap; five would need 120 KiB and fails to build. The epilogue shares this storage through a
// union, so it adds nothing to the budget.
// Declared only, like Fp8LinearA8ProductionSchedule above it: a geometry that has never been
// measured on this route must be a build error, not a silent opt-in to another problem's schedule.
template <class Geometry>
struct Fp8LinearA8TmaSchedule;

// 256-token tile, four stages. Four is the deepest that fits: 4 * (256 + 128) * 64 = 96 KiB against
// the 99 KiB per-CTA cap, and five would need 120 KiB. One CTA per SM follows from that budget.
using Fp8A8TmaMeasuredSchedule = Fp8A8TmaSchedule<256, 4, 1>;

template <>
struct Fp8LinearA8TmaSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

// Multiprocessor count of the device this thread will launch on - the current device, which is
// where the launch that follows goes. Cached per device ordinal: the count steers which kernel
// runs, and a process that touches two different GPUs must not steer the second one with the
// first one's number. Returning zero declines the route, which is always a safe answer.
inline std::int32_t fp8_a8_multiprocessor_count() {
    static std::array<std::atomic<std::int32_t>, kFp8A8MaxDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kFp8A8MaxDevices) {
        return 0;
    }
    const std::int32_t cached = cache[device].load(std::memory_order_acquire);
    if (cached != 0) { return cached; }
    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 0;
    }
    cache[device].store(value, std::memory_order_release);
    return value;
}

// Which of the two routes is cheaper at this width.
//
// Both kernels tile the same problem and both leave part of a wave idle at the end, but they
// quantise differently: the TMA route runs one CTA per SM over a 256-token tile, the cp.async route
// two CTAs per SM over a 64-token tile. So the decision is a comparison of two quantised costs, not
// a score for one of them - which also means the multiprocessor count enters the model on both
// sides instead of being frozen into a fitted constant.
//
// Cost of a route is (waves it needs) x (work one CTA does). Work per CTA is proportional to its
// token tile, so the tile widths carry it and cancel into kFp8A8TmaWorkRatio below.

// Time the TMA route takes per token of work, relative to the route it replaces. Solved from the
// widest measured point, where wave quantisation is mildest and the ratio is therefore closest to
// the kernels' intrinsic speeds: at 8192 tokens on 16384x5120 the measured ratio is 0.9555, which
// gives 0.936. It puts most measured points on the correct side of the comparison, but not
// all: six of them are wrong at every ratio in the legal range, and those are what the two
// width bounds below are for. The constant is a fit, not a law.
inline constexpr double kFp8A8TmaWorkRatio = 0.936;

// How much cheaper the model must find the TMA route before the route is taken. This is a margin on
// modelled cost, not on measured time: the model is a wave count times a tile width, so it is
// coarse, and a decision it calls within two percent is a decision it has not really made. Widths
// whose measured gain is smaller than this are still admitted - the margin buys confidence in the
// comparison, not a floor on the payoff.
inline constexpr double kFp8A8TmaMargin = 0.02;

// Which of the two routes the model calls cheaper, as a pure function of shape, width and part.
//
// This is the whole decision, and it is written once. The runtime guard below calls it after
// reading the multiprocessor count; the calibration underneath calls it with the count of the
// device the numbers were measured on. An earlier form had the calibration re-derive the same
// arithmetic from loose integers with every tile frozen as a default argument, which validated a
// copy: changing kBlockTokens or kMinBlocksPerSm in a production schedule would have left the
// assertions guarding a shape nothing instantiates.
template <class Geometry, class TmaSchedule, class MmaSchedule>
constexpr bool fp8_a8_tma_cheaper(std::int32_t tokens, std::int64_t multiprocessors) {
    const std::int64_t tma_blocks = fp8_a8_tma_blocks<Geometry, TmaSchedule>(tokens);
    const std::int64_t mma_tiles =
        (static_cast<std::int64_t>(tokens) + MmaSchedule::kBlockTokens - 1) /
        MmaSchedule::kBlockTokens;
    const std::int64_t mma_blocks =
        static_cast<std::int64_t>(Geometry::kOutputRows / MmaSchedule::kBlockRows) * mma_tiles;
    const std::int64_t mma_slots =
        static_cast<std::int64_t>(MmaSchedule::kMinBlocksPerSm) * multiprocessors;
    const std::int64_t tma_slots =
        static_cast<std::int64_t>(TmaSchedule::kMinBlocksPerSm) * multiprocessors;
    const std::int64_t tma_waves = (tma_blocks + tma_slots - 1) / tma_slots;
    const std::int64_t mma_waves = (mma_blocks + mma_slots - 1) / mma_slots;
    // Cost is waves times the work one SM carries through a wave - not waves alone. The two routes
    // put different amounts of work on an SM at once, and each route's occupancy comes from its own
    // schedule rather than from a number written here: the shipped TMA schedule places one CTA of
    // BlockM tokens, the cp.async schedule two CTAs of BlockTokens each. Comparing wave counts
    // without that weight makes the wider tile look free, which is exactly backwards - and reading
    // one route's occupancy from its schedule while fixing the other's in the model is how a future
    // schedule change silently stops being modelled.
    const double tma =
        static_cast<double>(tma_waves * TmaSchedule::kMinBlocksPerSm * TmaSchedule::kBlockM) *
        kFp8A8TmaWorkRatio;
    const double mma =
        static_cast<double>(mma_waves * MmaSchedule::kMinBlocksPerSm * MmaSchedule::kBlockTokens);
    return tma < mma * (1.0 - kFp8A8TmaMargin);
}

// The widest token count the route is allowed to take, per geometry.
//
// The cost model counts blocks along output rows and token tiles. Fp8Residual6144Geometry and
// Fp8Residual17408Geometry both have 5120 output rows, so it produces one verdict for both, and it
// has no term for K at all. Measured, they diverge: swept over nineteen widths from 1024 to 16384,
// 5120x17408 gains everywhere it is admitted, while 5120x6144 gains only up to 4096 and loses above
// it. 5120x6144 is also the smallest GEMM in the model - 2*N*K is 62.9 MFLOP per token against
// 146.8, 167.8, 178.3 and 356.5 for the other four - so it has the least work to amortise the
// pipeline over, which is the direction the numbers point in but not a mechanism this patch proves.
//
// So the bound is a measurement, not a model. It is stated per geometry rather than as a global
// rule so that a shape nobody swept cannot inherit a limit derived from this one.
template <class Geometry>
inline constexpr std::int32_t kFp8A8TmaMaxTokens = std::numeric_limits<std::int32_t>::max();
template <>
inline constexpr std::int32_t kFp8A8TmaMaxTokens<Fp8Residual6144Geometry> = 4096;

// The narrowest width the route is offered at, for all geometries.
//
// kFp8A8TmaWorkRatio was solved at the widths the product runs, 1024 and up. Below them the model
// is not merely uncalibrated, it is wrong about as often as it is right. Swept at every multiple of
// 64 from 256 to 1024 on all five shapes: attn_input measures 1.112 at 256 and 1.024 at 640 but
// 0.914 at 704 and 0.910 at 768; gdn_input measures 0.886 at 256 and 1.026 at 512; mlp_gate_up
// 1.040 at 256 and 0.940 at 768. The two residual shapes are the worst of the five: 5120x6144
// measures 1.164 to 1.177 across 576 to 768 and 5120x17408 measures 1.140 to 1.144 across the same
// widths, and both then turn to gains of 0.90 to 0.93 from 832 up, which the floor gives up.
//
// The absolute times say why, where the ratios do not. At 256 tokens the route takes 127.0 us on
// attn_input and 126.9 us on gdn_input - the same time for 14 percent more work - because with one
// token tile it is bound by the pipeline fill rather than by the work. So it wins exactly when the
// route it replaces is slower than that fill, which is a property of the other kernel's speed at
// that shape and not of anything this model counts. There is no monotone frontier here to put a
// per-shape floor on, so the floor goes where the calibration starts.
//
// Declined by this although measured faster: 0.886 on gdn_input at 256, 0.914 and 0.910 on
// attn_input at 704 and 768. Those widths arise only as the remainder chunk of a prompt, and giving
// them up is the cost of not shipping the losses beside them.
inline constexpr std::int32_t kFp8A8TmaMinTokens = 1024;

// The same question asked of a geometry, with both of its registered schedules filled in, and the
// measured width bound applied on top of the modelled cost.
template <class Geometry>
constexpr bool fp8_a8_tma_admits(std::int32_t tokens, std::int64_t multiprocessors) {
    if (tokens < kFp8A8TmaMinTokens) { return false; }
    if (tokens > kFp8A8TmaMaxTokens<Geometry>) { return false; }
    return fp8_a8_tma_cheaper<Geometry, typename Fp8LinearA8TmaSchedule<Geometry>::Type,
                              typename Fp8LinearA8ProductionSchedule<Geometry>::Type>(
        tokens, multiprocessors);
}

// The calibration, stated so that it cannot rot.
//
// kFp8A8TmaWorkRatio is a device constant: it is the TMA route's time per token of work relative to
// the route it replaces, and it was solved on an RTX 5090 (sm_120a, 170 SMs). It is NOT portable -
// on another part the two kernels' intrinsic speeds differ and this number would have to be
// re-solved. What travels is the shape of the comparison, not the constant.
//
// Below, every point measured on that device is written down as a compile-time check of the
// decision the model makes. These run through the cost model and the width bound, which is what the
// runtime consults for speed; the runtime then adds representability, the two tile-width gates and
// the grid.y limit on top, and none of those are exercised here. The declines are the ones
// that matter: each is a width where the TMA route was measured slower. Read them with care -
// only some are guarded by the model. Where a companion assertion records that the model alone
// would have admitted the width, the decline is the work of a width bound, and retuning the
// ratio or the margin will not disturb it. The ones without a companion are the ones that
// genuinely pin the constant.
inline constexpr std::int64_t kFp8A8CalibrationSms = 170;

// Measured slower - the model must decline these. Ratios are candidate over base.
// These three are declined by the floor. The model would take all of them, which is the whole
// reason the floor exists, so the companion assertions say so out loud: if the floor is ever
// removed on the belief that the model covers this range, these are what shows it does not.
static_assert(!fp8_a8_tma_admits<Fp8AttnInputGeometry>(256, kFp8A8CalibrationSms),
              "attn_input T=256 measured 1.112, must decline");
static_assert(fp8_a8_tma_cheaper<
                  Fp8AttnInputGeometry, typename Fp8LinearA8TmaSchedule<Fp8AttnInputGeometry>::Type,
                  typename Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry>::Type>(
                  256, kFp8A8CalibrationSms),
              "the model prefers TMA at 256 - the floor is what declines it");
static_assert(!fp8_a8_tma_admits<Fp8AttnInputGeometry>(640, kFp8A8CalibrationSms),
              "attn_input T=640 measured 1.024, must decline");
static_assert(!fp8_a8_tma_admits<Fp8GdnInputGeometry>(512, kFp8A8CalibrationSms),
              "gdn_input T=512 measured 1.026, must decline");
static_assert(!fp8_a8_tma_admits<Fp8GdnInputGeometry>(1024, kFp8A8CalibrationSms),
              "gdn_input T=1024 measured 1.115, must decline");
static_assert(!fp8_a8_tma_admits<Fp8GdnInputGeometry>(2048, kFp8A8CalibrationSms),
              "gdn_input T=2048 measured 1.031, must decline");

// Measured faster - the model must keep these.
static_assert(fp8_a8_tma_admits<Fp8GdnInputGeometry>(2560, kFp8A8CalibrationSms),
              "gdn_input T=2560 measured 0.959");
static_assert(fp8_a8_tma_admits<Fp8GdnInputGeometry>(4096, kFp8A8CalibrationSms),
              "gdn_input T=4096 measured 0.981");
static_assert(fp8_a8_tma_admits<Fp8GdnInputGeometry>(8192, kFp8A8CalibrationSms),
              "gdn_input T=8192 measured 0.956");
static_assert(fp8_a8_tma_admits<Fp8AttnInputGeometry>(4096, kFp8A8CalibrationSms),
              "attn_input T=4096 measured 0.951");
static_assert(fp8_a8_tma_admits<Fp8MlpGateUpGeometry>(4096, kFp8A8CalibrationSms),
              "mlp_gate_up T=4096 measured 0.949");
static_assert(fp8_a8_tma_admits<Fp8Residual17408Geometry>(1024, kFp8A8CalibrationSms),
              "residual 5120x17408 T=1024 measured 0.879");
static_assert(fp8_a8_tma_admits<Fp8Residual17408Geometry>(4096, kFp8A8CalibrationSms),
              "residual 5120x17408 T=4096 measured 0.917");
static_assert(fp8_a8_tma_admits<Fp8Residual6144Geometry>(1024, kFp8A8CalibrationSms),
              "residual 5120x6144 T=1024 measured 0.886");
static_assert(fp8_a8_tma_admits<Fp8Residual6144Geometry>(4096, kFp8A8CalibrationSms),
              "residual 5120x6144 T=4096 measured 0.928");

// The one shape whose bound is doing the work rather than the model. Ratios below are from
// ninfer_fp8_linear_add_bench, which is the call site these two geometries are reached from in
// production; the plain-linear numbers quoted above come from ninfer_linear_bench and are a
// different kernel with a different epilogue, so the two sets are not interchangeable.
//
// The first assertion records that the model, left alone, still prefers the TMA route at 5120. The
// second records what the width bound then does about it. If anyone removes the bound believing the
// model covers this, the first assertion is what tells them it does not.
static_assert(
    fp8_a8_tma_cheaper<Fp8Residual6144Geometry,
                       typename Fp8LinearA8TmaSchedule<Fp8Residual6144Geometry>::Type,
                       typename Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry>::Type>(
        5120, kFp8A8CalibrationSms),
    "the modelled cost still prefers TMA at 5120 - the width bound is what declines it");
static_assert(!fp8_a8_tma_admits<Fp8Residual6144Geometry>(5120, kFp8A8CalibrationSms),
              "residual 5120x6144 T=5120 measured 1.023, must decline");
static_assert(!fp8_a8_tma_admits<Fp8Residual6144Geometry>(8192, kFp8A8CalibrationSms),
              "residual 5120x6144 T=8192 measured 1.012, must decline");
static_assert(!fp8_a8_tma_admits<Fp8Residual6144Geometry>(12288, kFp8A8CalibrationSms),
              "residual 5120x6144 T=12288 measured 1.014, must decline");
// The neighbour the bound must not touch: same output rows, same block counts, nearly three times
// the K, and a gain at every one of those widths.
static_assert(fp8_a8_tma_admits<Fp8Residual17408Geometry>(8192, kFp8A8CalibrationSms),
              "residual 5120x17408 T=8192 measured 0.931");

// A width is only covered if the runtime would take it, and the runtime asks one thing the cost
// model does not: the width has to be a whole multiple of the tile of the route being replaced.
// kBlockTokens is the first template argument of the production schedule and the most tunable
// number in this file - raising it to 128 would drop 4160 at runtime while every assertion below
// stayed green, which is precisely the failure they exist to prevent. So coverage goes through
// both.
template <class Geometry>
constexpr bool fp8_a8_tma_covers(std::int32_t tokens) {
    using Mma = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    return fp8_a8_tma_admits<Geometry>(tokens, kFp8A8CalibrationSms) &&
           (tokens % Mma::kBlockTokens) == 0;
}

// These also constrain the ratio, and more tightly than some of the measurements do: solving the
// whole assertion block for kFp8A8TmaWorkRatio leaves a window whose upper end is set by
// attn_input at 4288, a coverage width, with under half a percent of headroom. So re-solving the
// constant on another part will most likely trip one of these first. The fix then is to re-pick
// the test width from that part's frontier and re-run the tests on it - not to widen the
// constant until the assertion passes, which would leave the tests covering nothing.
//
// The widths the numerical tests run at, asserted here so they cannot quietly stop covering the
// route. tests/ops/linear/test_fp8_a8.cpp and tests/ops/linear_add/test_fp8.cpp check values
// against a host reference; neither can see which kernel produced them, so if the predicate stopped
// admitting these widths the suite would stay green while the route it is there to cover
// disappeared. One aligned width and one that leaves a partial trailing tile per geometry, since
// the trailing tile is the majority of the admitted set and the part with the least coverage.
static_assert(fp8_a8_tma_covers<Fp8AttnInputGeometry>(4096) &&
                  fp8_a8_tma_covers<Fp8AttnInputGeometry>(4288),
              "test_fp8_a8.cpp covers attn_input at 4096 and 4288");
static_assert(fp8_a8_tma_covers<Fp8GdnInputGeometry>(4096) &&
                  fp8_a8_tma_covers<Fp8GdnInputGeometry>(4160),
              "test_fp8_a8.cpp covers gdn_input at 4096 and 4160");
static_assert(fp8_a8_tma_covers<Fp8MlpGateUpGeometry>(4096) &&
                  fp8_a8_tma_covers<Fp8MlpGateUpGeometry>(4288),
              "test_fp8_a8.cpp covers mlp_gate_up at 4096 and 4288");
static_assert(fp8_a8_tma_covers<Fp8Residual6144Geometry>(1664),
              "both fp8 tests cover residual 5120x6144 at 1664, its partial-tile width");
// attn_input reaches the route at the shipped default chunk, which is also a width its test
// runs. Every other route-taking test width is pinned below; without this one a retune could
// drop 1024 off the route and leave that test green over the kernel it was added to exercise.
static_assert(fp8_a8_tma_covers<Fp8AttnInputGeometry>(1024),
              "attn_input T=1024 is a route-taking test width");
static_assert(fp8_a8_tma_covers<Fp8Residual17408Geometry>(4160),
              "both fp8 tests cover residual 5120x17408 at 4160, its partial-tile width");

// The two residual geometries differ only in K, and the model has no K term: it counts blocks along
// output rows, of which both have 5120. So the pair of assertions above is one decision written
// twice, and it holds only because both shapes were measured at that width and both were gains.
// They are written out separately anyway: the day a schedule is registered per geometry is the day
// they stop agreeing, and the assertions should be the thing that notices.

// Guard for the TMA route.
//
// Eight conditions, in the order fp8_a8_tma_applies runs them, which is the order of what they
// protect: first what the hardware cannot describe, then what the calibration cannot speak
// for, then what the launch geometry cannot carry, and only last a question of speed.
//
// TMA descriptors carry a global address, and cuTensorMapEncodeTiled rejects one it cannot
// describe. The activation codes are a workspace this op allocated, so they are aligned by
// construction. The weight codes are a byte offset into the loaded artifact, so their alignment is
// a property of the layout and not something this patch establishes. Declining is the whole
// remedy - the route being replaced reads the same bytes with no such requirement - and it keeps a
// new failure mode out of a patch that is supposed to change only speed.
//
// This is a runtime address test and lives only here. fp8_a8_tma_admits and fp8_a8_tma_covers stay
// as they are: they are compile-time statements about widths, and an address is not a width.
inline constexpr std::uintptr_t kFp8A8TmaAddressAlignment = 16;

inline bool fp8_a8_tma_addresses_admit(const void* activation_codes, const void* weight_codes) {
    const auto a = reinterpret_cast<std::uintptr_t>(activation_codes);
    const auto b = reinterpret_cast<std::uintptr_t>(weight_codes);
    if (a == 0 || b == 0) { return false; }
    return (a % kFp8A8TmaAddressAlignment) == 0 && (b % kFp8A8TmaAddressAlignment) == 0;
}

// Representability is a property of the geometry. The branch below is documentation, not a
// fallback: the call sites test this predicate at runtime, so the kernel template is instantiated
// whatever it answers, and a geometry the kernel cannot tile fails its own static_assert first. A
// geometry is admitted to this route by being registered in Fp8LinearA8TmaSchedule above, and that
// list is declared-only for the same reason.
//
// The width must be a whole number of cp.async token tiles. This is what makes the route's output
// bit-identical rather than merely believed to be: at those widths the kernel it replaces takes its
// FullTokens branch, where the scaled accumulator feeds the epilogue directly - the same expression
// this kernel writes, and therefore the same rounding. At other widths the replaced kernel keeps
// the product live on a second path, which blocks the multiply-add contraction and rounds twice.
// One kernel shape cannot match both, so the route declines the widths it cannot match exactly.
//
// Then, and only then, is it a question of speed.
template <class Geometry, class TmaSchedule,
          class MmaSchedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type>
bool fp8_a8_tma_applies(std::int32_t tokens, const void* activation_codes,
                        const void* weight_codes) {
    // The width test below asks whether the route being replaced would take its FullTokens branch,
    // so it has to name that route's tile and no other. Pinning it here means a future per-op
    // schedule override cannot silently make the test ask about a tile nobody falls back to.
    static_assert(
        std::is_same_v<MmaSchedule, typename Fp8LinearA8ProductionSchedule<Geometry>::Type>,
        "the width test must name the schedule this route actually falls back to");
    if constexpr (!kFp8A8TmaRepresentable<Geometry, TmaSchedule>) {
        return false;
    } else {
        // Not a kernel requirement - the copy zero-fills past the extent and the store drops
        // those rows, so one partial tile is fine. It is a statement about the calibration: the
        // ratio was solved at and above kBlockM-sized widths, and a floor below the tile would put
        // the model somewhere it was never fitted.
        static_assert(kFp8A8TmaMinTokens >= TmaSchedule::kBlockM,
                      "the floor must stay inside the range the ratio was solved in");
        // Addresses before widths: a pointer the descriptor cannot describe is a hard failure in
        // the launcher, where a width that does not suit is only a slower route.
        if (!fp8_a8_tma_addresses_admit(activation_codes, weight_codes)) { return false; }
        if (tokens < kFp8A8TmaMinTokens) { return false; }
        if (tokens > kFp8A8TmaMaxTokens<Geometry>) { return false; }
        if ((tokens % MmaSchedule::kBlockTokens) != 0) { return false; }
        // The launcher puts token tiles on grid.y, which tops out at 65535 where the cp.async route
        // linearises into grid.x and does not. No real width comes near - it would take 16.7 M
        // tokens in one chunk - but the limit is the new route's alone, and exceeding it is a
        // launch failure rather than a slow answer, so it is cheaper to decline than to explain
        // later.
        if (fp8_a8_tma_token_tiles<Geometry, TmaSchedule>(tokens) > 65535) { return false; }
        const std::int32_t multiprocessors = fp8_a8_multiprocessor_count();
        if (multiprocessors <= 0) { return false; }
        return fp8_a8_tma_cheaper<Geometry, TmaSchedule, MmaSchedule>(tokens, multiprocessors);
    }
}

} // namespace ninfer::ops::detail
