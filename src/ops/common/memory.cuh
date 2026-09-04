#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

// Weight codes and their group scales are read once per decode round and never revisited, while the
// activation tile under them is re-read by every row of every CTA. An L2 eviction priority on the
// weight stream stops it evicting the working set it shares L2 with.
//
// These are written as PTX rather than through __ldcs because the two properties we want do not
// come in one intrinsic. The compiler already gives these __restrict__ weight pointers the
// non-coherent path (LDG.E.*.CONSTANT); __ldcs sets the eviction policy but drops it, so it trades
// one mechanism for the other instead of adding to it.
//
// The route is L2::cache_hint rather than the L2::evict_first modifier, because on sm_120a with
// CUDA 13.1 ptxas takes that modifier on ld only for the 256-bit forms:
//   ld.global[.nc].L2::evict_first.{u8,u16,u32,v4.b32}  -> rejected,
//     "Instruction 'ld' requires '.v8.b32/.v4.b64' type with '.L2::evict_first' modifier"
//   ld.global[.nc].L2::evict_first.v8.b32               -> accepted
//   createpolicy.fractional.L2::evict_first + ld.global.nc.L2::cache_hint.<any width> -> accepted
// The descriptor is one instruction and carries no state, so ptxas hoists it out of the loop.
__device__ __forceinline__ std::uint64_t l2_evict_first_policy() {
    std::uint64_t policy;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ std::uint16_t ld_nc_evict_first_u8(const void* ptr) {
    std::uint16_t value;
    asm("ld.global.nc.L2::cache_hint.u8 %0, [%1], %2;"
        : "=h"(value)
        : "l"(ptr), "l"(l2_evict_first_policy()));
    return value;
}

__device__ __forceinline__ std::uint16_t ld_nc_evict_first_u16(const void* ptr) {
    std::uint16_t value;
    asm("ld.global.nc.L2::cache_hint.u16 %0, [%1], %2;"
        : "=h"(value)
        : "l"(ptr), "l"(l2_evict_first_policy()));
    return value;
}

__device__ __forceinline__ std::uint32_t ld_nc_evict_first_u32(const void* ptr) {
    std::uint32_t value;
    asm("ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;"
        : "=r"(value)
        : "l"(ptr), "l"(l2_evict_first_policy()));
    return value;
}

__device__ __forceinline__ uint2 ld_nc_evict_first_v2(const void* ptr) {
    uint2 value;
    asm("ld.global.nc.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
        : "=r"(value.x), "=r"(value.y)
        : "l"(ptr), "l"(l2_evict_first_policy()));
    return value;
}

__device__ __forceinline__ uint4 ld_nc_evict_first_v4(const void* ptr) {
    uint4 value;
    asm("ld.global.nc.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(ptr), "l"(l2_evict_first_policy()));
    return value;
}

template <class V, class T>
__device__ __forceinline__ V load_vec_streaming(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    const void* source = static_cast<const void*>(ptr);
    if constexpr (sizeof(V) == 1) {
        const std::uint16_t raw = ld_nc_evict_first_u8(source);
        const std::uint8_t narrowed = static_cast<std::uint8_t>(raw);
        return *reinterpret_cast<const V*>(&narrowed);
    } else if constexpr (sizeof(V) == 2) {
        const std::uint16_t raw = ld_nc_evict_first_u16(source);
        return *reinterpret_cast<const V*>(&raw);
    } else if constexpr (sizeof(V) == 4) {
        const std::uint32_t raw = ld_nc_evict_first_u32(source);
        return *reinterpret_cast<const V*>(&raw);
    } else if constexpr (sizeof(V) == 8) {
        const uint2 raw = ld_nc_evict_first_v2(source);
        return *reinterpret_cast<const V*>(&raw);
    } else {
        const uint4 raw = ld_nc_evict_first_v4(source);
        return *reinterpret_cast<const V*>(&raw);
    }
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
