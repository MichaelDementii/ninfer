#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_tma.cuh"
#include "ops/linear/fp8/fp8_config.h"

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

// TMA route. One CTA per SM with a 256-token tile; three stages keep the shared budget at 72 KiB,
// which still leaves room for the epilogue staging buffer that shares the same storage.
template <class Geometry>
struct Fp8LinearA8TmaSchedule {
    using Type = Fp8A8TmaSchedule<256, 4, 1>;
};

// Guard for the TMA route: at least one token tile, and both weight extents divisible by the
// block tile. A partial trailing tile is allowed; TMA zero-fills it and the epilogue skips it.
template <class Geometry, class Schedule>
bool fp8_a8_tma_applies(std::int32_t tokens) {
    if (tokens < Schedule::kBlockM) { return false; }
    if ((Geometry::kOutputRows % Schedule::kBlockN) != 0) { return false; }
    if ((Geometry::kInputRows % Schedule::kBlockK) != 0) { return false; }
    return true;
}

} // namespace ninfer::ops::detail
