"""Resolve the three conflicts from cherry-picking 9c67c692 onto 3a61ef3f.

Master's fix (3a61ef3f) split what used to be one buffer into two: select_count writes the
tile-local rank into `local_rank`, and gather publishes the inverse map into `packed_index`, which
is aliased onto the dead prefix of `tile_counts`. Our change replaces gather with an index kernel
that publishes the map without moving activations, so it reads the rank from `local_rank` now
rather than from `packed_index` in place - which also means the read-write overlap master's comment
warns about does not arise on this path at all.
"""
import pathlib, sys

H = pathlib.Path("/root/ninfer_d4/src/ops/sparse_moe/prefill/sparse_moe_prefill.h")
C = pathlib.Path("/root/ninfer_d4/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu")

h = H.read_text()
c = C.read_text()

# ---- .h, conflict 1: keep master's local_rank, add our packed_token ----------------------------
old = """<<<<<<< HEAD
    // Selection writes one rank local to a routing tile. Gather must keep this source separate
    // from the final inverse map because all threads in an assignment block consume the rank while
    // thread 0 publishes the packed column.
    Tensor local_rank;
=======
    // Selection first writes a rank local to one routing tile. Gather replaces
    // it in place with the inverse map from token/route slot to packed column.
    Tensor packed_index;
    // Inverse of packed_index: the token each packed column was routed from. Lets the routed
    // gate/up GEMM stage its activation tile from `x` instead of a materialised copy.
    Tensor packed_token;
>>>>>>> 9c67c692 (perf(ops): stage the routed MoE activations from x instead of a gathered copy)
"""
new = """    // Selection writes one rank local to a routing tile. Gather must keep this source separate
    // from the final inverse map because all threads in an assignment block consume the rank while
    // thread 0 publishes the packed column.
    Tensor local_rank;
    // Inverse of packed_index: the token each packed column was routed from. Lets the routed
    // gate/up GEMM stage its activation tile from `x` instead of a materialised copy.
    Tensor packed_token;
"""
assert h.count(old) == 1, "h conflict 1 anchor"
h = h.replace(old, new)

# ---- .h, conflict 2: keep master's allocation, add packed_token --------------------------------
old = """<<<<<<< HEAD
    out.local_rank     = arena.alloc(DType::I32, {assignments}, 256);
=======
    out.packed_index   = arena.alloc(DType::I32, {assignments}, 256);
    out.packed_token   = arena.alloc(DType::I32, {assignments}, 256);
>>>>>>> 9c67c692 (perf(ops): stage the routed MoE activations from x instead of a gathered copy)
"""
new = """    out.local_rank     = arena.alloc(DType::I32, {assignments}, 256);
    out.packed_token   = arena.alloc(DType::I32, {assignments}, 256);
"""
assert h.count(old) == 1, "h conflict 2 anchor"
h = h.replace(old, new)

# ---- .cu, conflict 3: both launch sites take local_rank as the rank source ---------------------
old = """<<<<<<< HEAD
            sparse_moe_prefill_gather_kernel<true><<<assignments, kExpertThreads, 0, stream>>>(
                input, ids, local_rank, packed_index, tile_bases, grouped_io, route_job_count);
=======
            if (routed_gate_up_q4) {
                sparse_moe_prefill_index_kernel<true><<<index_blocks, kExpertThreads, 0, stream>>>(
                    ids, packed_index, tile_bases, packed_token, assignments, route_job_count);
            } else {
                sparse_moe_prefill_gather_kernel<true><<<assignments, kExpertThreads, 0, stream>>>(
                    input, ids, packed_index, tile_bases, grouped_io, route_job_count);
            }
        } else if (routed_gate_up_q4) {
            sparse_moe_prefill_index_kernel<false><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, packed_index, tile_bases, packed_token, assignments, nullptr);
>>>>>>> 9c67c692 (perf(ops): stage the routed MoE activations from x instead of a gathered copy)
"""
new = """            if (routed_gate_up_q4) {
                sparse_moe_prefill_index_kernel<true><<<index_blocks, kExpertThreads, 0, stream>>>(
                    ids, local_rank, packed_index, tile_bases, packed_token, assignments,
                    route_job_count);
            } else {
                sparse_moe_prefill_gather_kernel<true><<<assignments, kExpertThreads, 0, stream>>>(
                    input, ids, local_rank, packed_index, tile_bases, grouped_io, route_job_count);
            }
        } else if (routed_gate_up_q4) {
            sparse_moe_prefill_index_kernel<false><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, local_rank, packed_index, tile_bases, packed_token, assignments, nullptr);
"""
assert c.count(old) == 1, "cu conflict anchor"
c = c.replace(old, new)

# ---- the index kernel now reads the rank from local_rank, not from packed_index in place -------
old = """__global__ void sparse_moe_prefill_index_kernel(const int* __restrict__ ids,
                                                int* __restrict__ packed_index,
                                                const int* __restrict__ tile_bases,
                                                int* __restrict__ packed_token, int assignments,
                                                const int* __restrict__ route_job_count) {"""
new = """__global__ void sparse_moe_prefill_index_kernel(const int* __restrict__ ids,
                                                const int* __restrict__ local_rank,
                                                int* __restrict__ packed_index,
                                                const int* __restrict__ tile_bases,
                                                int* __restrict__ packed_token, int assignments,
                                                const int* __restrict__ route_job_count) {"""
assert c.count(old) == 1, "index kernel signature anchor"
c = c.replace(old, new)

old = """    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + packed_index[assignment];
    packed_index[assignment] = packed;
    packed_token[packed]     = token;"""
new = """    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_token[packed]     = token;"""
assert c.count(old) == 1, "index kernel body anchor"
c = c.replace(old, new)

# the comment above it should say where the rank comes from now
old = "// Publishes the packed-column map without moving activations. One thread per assignment."
new = ("// Publishes the packed-column map without moving activations. One thread per assignment, so\n"
       "// unlike the gather it reads the tile-local rank and writes the inverse map from the same\n"
       "// thread and never has the two live in one buffer.")
assert c.count(old) == 1, "index kernel comment anchor"
c = c.replace(old, new)

H.write_text(h)
C.write_text(c)
left = [l for f in (h, c) for l in f.splitlines() if l.startswith(("<<<<<<<", "=======", ">>>>>>>"))]
print("resolved; leftover conflict markers:", len(left))
sys.exit(1 if left else 0)
