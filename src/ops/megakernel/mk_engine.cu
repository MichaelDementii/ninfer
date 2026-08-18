#define NINFER_MK_PROF 1
#define NINFER_MK_ENGINE 1

#include "ops/megakernel/mk_engine.h"

#include "ops/megakernel/mk_core.cuh"
#include "ops/megakernel/mk_instr.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::ops::mk {
namespace {

constexpr int kRowSlice   = 16;
constexpr int kInSlices   = 12288 / kRowSlice;   // 768
constexpr int kHeadSlices = 8192 / kRowSlice;    // 512
constexpr int kOutSlices  = 2048 / kRowSlice;    // 128
constexpr int kRecSlices  = (32 * 32) / 16;      // 64
constexpr int kD1Slices   = 129;                 // 2 router rows per slice
constexpr int kD3SharedSlices = 512 / 16;        // 32: shared path, no ids needed
constexpr int kD3RoutedSlices = (8 * 512) / 16;  // 256: 16 consecutive j of one path
constexpr int kD4Slices   = 2048 / 16;           // 128: path-major rounds, warp per row

#define MK_CHECK(call)                                                                   \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "megakernel engine CUDA error %s at %s:%d\n",           \
                         cudaGetErrorString(err_), __FILE__, __LINE__);                  \
            throw std::runtime_error("megakernel engine CUDA failure");                  \
        }                                                                                \
    } while (0)

std::int64_t pack_f32(float v) {
    int bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<std::int64_t>(bits);
}

struct Workspace {
    __nv_bfloat16* h    = nullptr;   // 2048
    __nv_bfloat16* h2   = nullptr;   // 2048
    __nv_bfloat16* qc   = nullptr;   // 2048
    __nv_bfloat16* kc   = nullptr;   // 2048
    __nv_bfloat16* vc   = nullptr;   // 4096
    __nv_bfloat16* z    = nullptr;   // 4096
    __nv_bfloat16* o    = nullptr;   // 4096
    __nv_bfloat16* on   = nullptr;   // 4096
    float* ab           = nullptr;   // 64
    float* gf           = nullptr;   // 32
    float* betaf        = nullptr;   // 32
    float* scores       = nullptr;   // 257
    float* alpha        = nullptr;   // 8
    float* sscale       = nullptr;   // 1
    float* act          = nullptr;   // 9*512
    int* ids            = nullptr;   // 8
    float* fgp          = nullptr;   // fused gating partials: 32x64 a/b + 32 norm
};

struct Recorder {
    bool ready          = false;
    bool prof           = false;
    int rounds_done     = 0;
    unsigned long long* wait_ns = nullptr;
    unsigned long long* exec_ns = nullptr;
    // prof v2: [0..cap) = per-tape-index min work-start (globaltimer ns),
    // [cap..2cap) = max end; seg_stamp[seg]=min entry, [32+seg]=max exit.
    unsigned long long* span      = nullptr;
    unsigned long long* seg_stamp = nullptr;
    std::vector<MkOp> tape_ops;   // host mirror: op per tape index
    int n_sm            = 0;
    Workspace ws{};
    std::uint32_t* counters = nullptr;
    std::uint32_t counter_capacity = 1u << 16;
    std::uint32_t ctr_next  = 0;

    std::vector<MkInstr> pending;      // host staging for the current segment
    MkInstr* tape_device    = nullptr; // persistent arena for all captured tapes
    std::size_t tape_capacity = 0;
    std::size_t tape_used     = 0;
    MkStream* streams_device  = nullptr;   // arena of per-segment stream arrays
    std::size_t streams_capacity = 0;
    std::size_t streams_used     = 0;
    // Tape content is identical every round (same pointers, same counter indices):
    // dedup uploads by content so replay rounds and graph capture reuse the device
    // segment uploaded on the first eager round — no H2D memcpy nodes in the graph.
    struct SegmentSlot {
        MkInstr* tape;
        MkStream* streams;
    };
    std::unordered_map<std::string, SegmentSlot> segment_cache;

    // per-round chaining state
    std::uint32_t prev_out_ctr = kMkNone;
    std::uint32_t prev_out_cnt = 0;
    std::uint32_t prev_in_ctr  = kMkNone;
    std::uint32_t prev_in_cnt  = 0;
    std::uint32_t prev_d4_ctr  = kMkNone;
    std::uint32_t prev_d4_cnt  = 0;
    int segments_this_round    = 0;
    int classes_this_round     = 0;
    bool logged_once           = false;

    void init() {
        if (ready) { return; }
        cudaDeviceProp prop{};
        MK_CHECK(cudaGetDeviceProperties(&prop, 0));
        n_sm = prop.multiProcessorCount;
        MK_CHECK(cudaMalloc(&ws.h, 2048 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.h2, 2048 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.qc, 2048 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.kc, 2048 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.vc, 4096 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.z, 4096 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.o, 4096 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.on, 4096 * sizeof(__nv_bfloat16)));
        MK_CHECK(cudaMalloc(&ws.ab, 64 * sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.gf, 32 * sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.betaf, 32 * sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.scores, 257 * sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.alpha, 8 * sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.sscale, sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.act, 9 * 512 * sizeof(float)));
        MK_CHECK(cudaMalloc(&ws.ids, 8 * sizeof(int)));
        MK_CHECK(cudaMalloc(&ws.fgp, (32 * 64 + 32) * sizeof(float)));
        MK_CHECK(cudaMalloc(&counters, counter_capacity * sizeof(std::uint32_t)));
        // Preallocate the tape/stream arenas: cudaMalloc is forbidden while a
        // stream is capturing, and flushes happen inside graph capture.
        tape_capacity = 1u << 16;   // 64K instructions (~150 rounds)
        MK_CHECK(cudaMalloc(&tape_device, tape_capacity * sizeof(MkInstr)));
        streams_capacity = 256 * static_cast<std::size_t>(n_sm);
        MK_CHECK(cudaMalloc(&streams_device, streams_capacity * sizeof(MkStream)));
        const char* prof_env = std::getenv("NINFER_MEGAKERNEL_PROF");
        prof                 = prof_env != nullptr && prof_env[0] == '1';
        if (prof) {
            MK_CHECK(cudaMalloc(&wait_ns, tape_capacity * sizeof(unsigned long long)));
            MK_CHECK(cudaMalloc(&exec_ns, tape_capacity * sizeof(unsigned long long)));
            MK_CHECK(cudaMemset(wait_ns, 0, tape_capacity * sizeof(unsigned long long)));
            MK_CHECK(cudaMemset(exec_ns, 0, tape_capacity * sizeof(unsigned long long)));
            MK_CHECK(cudaMalloc(&span, 2 * tape_capacity * sizeof(unsigned long long)));
            MK_CHECK(cudaMalloc(&seg_stamp, 64 * sizeof(unsigned long long)));
        }
        ready = true;
    }

    MkInstr blank() {
        MkInstr instr{};
        instr.op            = MkOp::Noop;
        instr.task_counter  = kMkNone;
        instr.slice_count   = 1;
        instr.done_counter  = kMkNone;
        instr.done2_counter = kMkNone;
        for (int w = 0; w < kMkMaxWaits; ++w) { instr.wait_counter[w] = kMkNone; }
        return instr;
    }

    std::uint32_t alloc_ctr() {
        if (ctr_next >= counter_capacity) {
            throw std::runtime_error("megakernel: counter arena exhausted");
        }
        return ctr_next++;
    }

    void push(MkInstr instr) {
        instr.task_counter = alloc_ctr();
        pending.push_back(instr);
        // Host mirror of the tape arena: pushes land in arena order across ALL
        // rounds (eager + capture), so profiling can attribute every offset —
        // replays accumulate into the CAPTURE round's tape region.
        tape_ops.push_back(instr.op);
        ++classes_this_round;
    }
};

Recorder& rec() {
    static Recorder r;
    return r;
}

void prof_dump_at_exit() {
    Recorder& r = rec();
    if (r.wait_ns == nullptr) { return; }
    cudaDeviceSynchronize();
    const std::size_t n = r.tape_ops.size();
    if (n == 0) { return; }
    std::vector<unsigned long long> wait_h(n);
    std::vector<unsigned long long> exec_h(n);
    cudaMemcpy(wait_h.data(), r.wait_ns, n * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(exec_h.data(), r.exec_ns, n * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    double ws[32] = {};
    double es[32] = {};
    for (std::size_t i = 0; i < n; ++i) {
        const int op = static_cast<int>(r.tape_ops[i]) & 31;
        ws[op] += static_cast<double>(wait_h[i]);
        es[op] += static_cast<double>(exec_h[i]);
    }
    std::fprintf(stderr,
                 "[megakernel prof atexit] per-op CTA-clock sums (us at 2.4GHz), "
                 "tape_ops=%zu tape_used=%zu:\n",
                 n, r.tape_used);
    for (int op = 0; op < 32; ++op) {
        if (ws[op] + es[op] == 0.0) { continue; }
        std::fprintf(stderr, "  op%-2d wait %14.0f exec %14.0f\n", op, ws[op] / 2400.0,
                     es[op] / 2400.0);
    }
    // prof v2: last round's wall timeline (globaltimer ns, relative to round base).
    if (r.span != nullptr) {
        std::vector<unsigned long long> smin(r.tape_capacity), smax(r.tape_capacity);
        std::vector<unsigned long long> seg(64);
        cudaMemcpy(smin.data(), r.span, r.tape_capacity * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(smax.data(), r.span + r.tape_capacity,
                   r.tape_capacity * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(seg.data(), r.seg_stamp, 64 * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        unsigned long long base = ~0ull;
        for (int s = 0; s < 32; ++s) {
            if (seg[s] != ~0ull && seg[s] < base) { base = seg[s]; }
        }
        for (std::size_t i = 0; i < n && i < r.tape_capacity; ++i) {
            if (smin[i] != ~0ull && smin[i] < base) { base = smin[i]; }
        }
        std::fprintf(stderr, "[mk prof spans] base_ns=%llu rounds=%d\n", base, r.rounds_done);
        for (int s = 0; s < 32; ++s) {
            if (seg[s] == ~0ull || seg[32 + s] == 0) { continue; }
            std::fprintf(stderr, "SEG %-2d in %9.2f out %9.2f\n", s,
                         (seg[s] - base) / 1000.0, (seg[32 + s] - base) / 1000.0);
        }
        for (std::size_t i = 0; i < n && i < r.tape_capacity; ++i) {
            if (smin[i] == ~0ull || smax[i] == 0) { continue; }
            std::fprintf(stderr, "CLS %-4zu op%-2d start %9.2f end %9.2f\n", i,
                         static_cast<int>(r.tape_ops[i]) & 31, (smin[i] - base) / 1000.0,
                         (smax[i] - base) / 1000.0);
        }
    }
}

} // namespace

bool mk_engine_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_MEGAKERNEL");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

void mk_begin_round(cudaStream_t stream) {
    Recorder& r = rec();
    r.init();
    if (r.prof) {
        static const bool registered = [] {
            std::atexit(prof_dump_at_exit);
            return true;
        }();
        (void)registered;
    }
    r.ctr_next            = 0;
    r.prev_out_ctr        = kMkNone;
    r.prev_in_ctr         = kMkNone;
    r.prev_d4_ctr         = kMkNone;
    r.segments_this_round = 0;
    r.classes_this_round  = 0;
    r.pending.clear();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    MK_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status == cudaStreamCaptureStatusNone) {
        // Eager rounds (e.g. --no-cuda-graph profiling) can reuse the arenas: the
        // prior round has fully drained by the next host-side decode step.
        r.tape_used    = 0;
        r.streams_used = 0;
    }
    MK_CHECK(cudaMemsetAsync(r.counters, 0, r.counter_capacity * sizeof(std::uint32_t), stream));
    if (r.prof) {
        // Captured as graph nodes: every round (replays included) resets the span
        // arrays, so post-run reads see the LAST round's wall timeline.
        MK_CHECK(cudaMemsetAsync(r.span, 0xff, r.tape_capacity * sizeof(unsigned long long),
                                 stream));
        MK_CHECK(cudaMemsetAsync(r.span + r.tape_capacity, 0,
                                 r.tape_capacity * sizeof(unsigned long long), stream));
        MK_CHECK(cudaMemsetAsync(r.seg_stamp, 0xff, 32 * sizeof(unsigned long long), stream));
        MK_CHECK(cudaMemsetAsync(r.seg_stamp + 32, 0, 32 * sizeof(unsigned long long), stream));
    }
}

void mk_record_gdn_mixer(const MkGdnMixerArgs& a) {
    Recorder& r = rec();

    // Author's fused norm+gating (two-phase transplant) vs the serial
    // norm -> a/b gemv chain. The fused path reproduces the engine's h and
    // g/beta bit-for-bit (split-32 ordered reductions).
    static const bool fused_gating = [] {
        const char* env = std::getenv("NINFER_MK_FUSED_GATING");
        return env == nullptr || env[0] != '0';
    }();

    std::uint32_t c_norm = kMkNone;   // fused: FGB counter gates h consumers
    std::uint32_t c_ga   = kMkNone;
    std::uint32_t c_gb   = kMkNone;

    if (fused_gating) {
        MkInstr fga      = r.blank();
        fga.op           = MkOp::FusedGateA;
        fga.ptr[0]       = a.x;
        fga.ptr[1]       = a.input_norm_w;
        fga.ptr[2]       = a.a_w;
        fga.ptr[3]       = a.b_w;
        fga.out[0]       = r.ws.fgp;
        fga.dim[0]       = 0;
        fga.dim[1]       = 1;
        fga.dim[6]       = 1;
        fga.slice_count  = 64;   // (split 0..31) x (head tile 0..1)
        fga.done_counter = r.alloc_ctr();
        if (r.prev_d4_ctr != kMkNone) {
            fga.wait_counter[0] = r.prev_d4_ctr;
            fga.wait_target[0]  = r.prev_d4_cnt;
        }
        if (r.prev_in_ctr != kMkNone) {
            fga.wait_counter[1] = r.prev_in_ctr;
            fga.wait_target[1]  = r.prev_in_cnt;
        }
        const std::uint32_t c_fga = fga.done_counter;
        r.push(fga);

        MkInstr fgb         = r.blank();
        fgb.op              = MkOp::FusedGateB;
        fgb.ptr[0]          = a.x;
        fgb.ptr[1]          = a.input_norm_w;
        fgb.ptr[2]          = r.ws.fgp;
        fgb.ptr[3]          = a.a_log;
        fgb.ptr[4]          = a.dt_bias;
        fgb.ptr[5]          = r.ws.betaf;
        fgb.out[0]          = r.ws.h;
        fgb.out[1]          = r.ws.gf;
        fgb.dim[1]          = pack_f32(a.rms_eps);
        fgb.done_counter    = r.alloc_ctr();
        fgb.wait_counter[0] = c_fga;
        fgb.wait_target[0]  = 64;
        c_norm              = fgb.done_counter;   // gates h AND g/beta
        r.push(fgb);
    } else {
        MkInstr norm      = r.blank();
        norm.op           = MkOp::RmsNorm2048;
        norm.ptr[0]       = a.x;
        norm.ptr[1]       = a.input_norm_w;
        norm.out[0]       = r.ws.h;
        norm.dim[0]       = 1;
        norm.dim[1]       = pack_f32(a.rms_eps);
        norm.done_counter = r.alloc_ctr();
        if (r.prev_d4_ctr != kMkNone) {
            norm.wait_counter[0] = r.prev_d4_ctr;
            norm.wait_target[0]  = r.prev_d4_cnt;
        }
        if (r.prev_in_ctr != kMkNone) {
            norm.wait_counter[1] = r.prev_in_ctr;
            norm.wait_target[1]  = r.prev_in_cnt;
        }
        c_norm = norm.done_counter;
        r.push(norm);

        MkInstr ga         = r.blank();
        ga.op              = MkOp::Bf16Gemv;
        ga.ptr[0]          = r.ws.h;
        ga.ptr[1]          = a.a_w;
        ga.out[0]          = r.ws.ab;
        ga.dim[0]          = 0;
        ga.dim[1]          = 32;
        ga.dim[2]          = 2048;
        ga.dim[3]          = 1;
        ga.dim[6]          = 1;
        ga.done_counter    = r.alloc_ctr();
        ga.wait_counter[0] = c_norm;
        ga.wait_target[0]  = 1;
        c_ga               = ga.done_counter;
        r.push(ga);

        MkInstr gb         = r.blank();
        gb.op              = MkOp::Bf16Gemv;
        gb.ptr[0]          = r.ws.h;
        gb.ptr[1]          = a.b_w;
        gb.out[0]          = r.ws.ab + 32;
        gb.dim[0]          = 0;
        gb.dim[1]          = 32;
        gb.dim[2]          = 2048;
        gb.dim[3]          = 1;
        gb.dim[6]          = 1;
        gb.done_counter    = r.alloc_ctr();
        gb.wait_counter[0] = c_norm;
        gb.wait_target[0]  = 1;
        c_gb               = gb.done_counter;
        r.push(gb);
    }

    MkInstr in         = r.blank();
    in.op              = MkOp::W8DecodeConv;
    in.ptr[0]          = r.ws.h;
    in.ptr[1]          = a.qkvz_codes;
    in.ptr[2]          = a.qkvz_scales;
    in.ptr[3]          = a.conv_w;
    in.ptr[4]          = a.conv_state;
    in.ptr[5]          = r.ws.vc;
    in.ptr[6]          = r.ws.z;
    in.ptr[7]          = a.state_slots;
    in.out[0]          = r.ws.qc;
    in.out[1]          = r.ws.kc;
    in.dim[0]          = 0;
    in.dim[1]          = 2 * kRowSlice;
    in.slice_count     = kInSlices / 2;
    in.done_counter    = r.alloc_ctr();
    in.done2_counter   = r.alloc_ctr();
    in.done2_limit     = kHeadSlices / 2;
    in.wait_counter[0] = c_norm;
    in.wait_target[0]  = 1;
    const std::uint32_t c_in      = in.done_counter;
    const std::uint32_t c_in_head = in.done2_counter;
    r.push(in);
    r.prev_in_ctr = c_in;
    r.prev_in_cnt = kInSlices / 2;

    // Fused path: rec reads FGB's engine-exact g/beta arrays directly.
    // Legacy path: gating transform folded into the recurrent body (dim[7]=1)
    // — no gg class, rec gated by the in-proj head chunk + the tiny a/b gemvs.
    MkInstr rc = r.blank();
    rc.op      = MkOp::GdnRecurrent;
    rc.ptr[0]  = r.ws.qc;
    rc.ptr[1]  = r.ws.kc;
    rc.ptr[2]  = r.ws.vc;
    if (fused_gating) {
        rc.ptr[3] = r.ws.gf;
        rc.ptr[4] = r.ws.betaf;
        rc.dim[7] = 0;
    } else {
        rc.ptr[3] = r.ws.ab;
        rc.ptr[4] = a.a_log;
        rc.ptr[7] = a.dt_bias;
        rc.dim[7] = 1;
    }
    rc.ptr[5]          = a.state_slots;
    rc.ptr[6]          = a.state_slots;
    rc.out[0]          = r.ws.o;
    rc.out[1]          = a.rec_state;
    rc.dim[0]          = 0;
    rc.dim[1]          = 16;
    rc.dim[2]          = pack_f32(a.gdn_scale);
    rc.dim[3]          = a.rec_slot_stride;
    rc.dim[6]          = 1;
    rc.slice_count     = kRecSlices;
    rc.done_counter    = r.alloc_ctr();
    rc.wait_counter[0] = c_in_head;
    rc.wait_target[0]  = kHeadSlices / 2;
    if (fused_gating) {
        // c_norm (= FGB) already gates in+conv upstream; g/beta share it.
        rc.wait_counter[1] = c_norm;
        rc.wait_target[1]  = 1;
    } else {
        rc.wait_counter[1] = c_ga;
        rc.wait_target[1]  = 1;
        rc.wait_counter[2] = c_gb;
        rc.wait_target[2]  = 1;
    }
    const std::uint32_t c_rec = rc.done_counter;
    r.push(rc);

    MkInstr gn         = r.blank();
    gn.op              = MkOp::GatedNorm128;
    gn.ptr[0]          = r.ws.o;
    gn.ptr[1]          = a.gdn_norm_w;
    gn.ptr[2]          = r.ws.z;
    gn.out[0]          = r.ws.on;
    gn.dim[0]          = 0;
    gn.dim[1]          = 32;
    gn.dim[2]          = pack_f32(a.rms_eps);
    gn.done_counter    = r.alloc_ctr();
    gn.wait_counter[0] = c_rec;
    gn.wait_target[0]  = kRecSlices;
    gn.wait_counter[1] = c_in;
    gn.wait_target[1]  = kInSlices / 2;
    const std::uint32_t c_gn = gn.done_counter;
    r.push(gn);

    MkInstr out         = r.blank();
    out.op              = MkOp::W8DecodeK;
    out.ptr[0]          = r.ws.on;
    out.ptr[1]          = a.out_codes;
    out.ptr[2]          = a.out_scales;
    out.out[0]          = a.x;
    out.dim[0]          = 0;
    out.dim[1]          = kRowSlice;
    out.dim[2]          = 4096;
    out.dim[3]          = 1;
    out.dim[6]          = 1;
    out.slice_count     = kOutSlices;
    out.done_counter    = r.alloc_ctr();
    out.wait_counter[0] = c_gn;
    out.wait_target[0]  = 1;
    r.push(out);
    r.prev_out_ctr = out.done_counter;
    r.prev_out_cnt = kOutSlices;
}

void mk_record_moe(const MkMoeArgs& a) {
    Recorder& r = rec();

    MkInstr norm2      = r.blank();
    norm2.op           = MkOp::RmsNorm2048;
    norm2.ptr[0]       = a.x;
    norm2.ptr[1]       = a.post_norm_w;
    norm2.out[0]       = r.ws.h2;
    norm2.dim[0]       = 1;
    norm2.dim[1]       = pack_f32(a.rms_eps);
    norm2.done_counter = r.alloc_ctr();
    if (r.prev_out_ctr != kMkNone) {
        norm2.wait_counter[0] = r.prev_out_ctr;
        norm2.wait_target[0]  = r.prev_out_cnt;
    }
    const std::uint32_t c_norm2 = norm2.done_counter;
    r.push(norm2);

    MkInstr d1         = r.blank();
    d1.op              = MkOp::MoeD1;
    d1.ptr[0]          = r.ws.h2;
    d1.ptr[1]          = a.router;
    d1.out[0]          = r.ws.scores;
    d1.dim[0]          = 0;
    d1.dim[1]          = 2;
    d1.dim[6]          = 1;
    d1.slice_count     = kD1Slices;
    d1.done_counter    = r.alloc_ctr();
    d1.wait_counter[0] = c_norm2;
    d1.wait_target[0]  = 1;
    const std::uint32_t c_d1 = d1.done_counter;
    r.push(d1);

    // Shared-expert gate_up needs only h2 — it streams its 2MB while d1/d2
    // (router + single-warp top-8) would otherwise leave the bus idle.
    MkInstr d3s         = r.blank();
    d3s.op              = MkOp::MoeD3;
    d3s.ptr[0]          = r.ws.h2;
    d3s.ptr[5]          = a.sgu_codes;
    d3s.ptr[6]          = a.sgu_scales;
    d3s.out[0]          = r.ws.act;
    d3s.dim[0]          = 0;
    d3s.dim[1]          = 16;
    d3s.dim[3]          = 1;   // shared-only class
    d3s.dim[6]          = 1;
    d3s.slice_count     = kD3SharedSlices;
    d3s.done_counter    = r.alloc_ctr();
    d3s.wait_counter[0] = c_norm2;
    d3s.wait_target[0]  = 1;
    const std::uint32_t c_d3s = d3s.done_counter;
    r.push(d3s);

    // d3 consumes ONLY ids[]; alpha/sscale are d4 inputs. With the d2 split the
    // ids land on done2 mid-body (right after the selection loop), so d3 starts
    // while d2's single warp still runs softmax + shared-scale.
    static const bool d2_split = [] {
        const char* env = std::getenv("NINFER_MK_D2SPLIT");
        return env == nullptr || env[0] != '0';
    }();

    MkInstr d2         = r.blank();
    d2.op              = MkOp::MoeD2;
    d2.ptr[0]          = r.ws.scores;
    d2.ptr[1]          = r.ws.sscale;
    d2.out[0]          = r.ws.ids;
    d2.out[1]          = r.ws.alpha;
    d2.done_counter    = r.alloc_ctr();
    if (d2_split) {
        d2.done2_counter = r.alloc_ctr();
        d2.done2_limit   = 0;   // interpreter never posts it; the body does
    }
    d2.wait_counter[0] = c_d1;
    d2.wait_target[0]  = kD1Slices;
    const std::uint32_t c_d2     = d2.done_counter;
    const std::uint32_t c_d2_ids = d2_split ? d2.done2_counter : d2.done_counter;
    r.push(d2);

    MkInstr d3         = r.blank();
    d3.op              = MkOp::MoeD3;
    d3.ptr[0]          = r.ws.h2;
    d3.ptr[1]          = r.ws.ids;
    d3.ptr[2]          = a.rgu_codes;
    d3.ptr[4]          = a.rgu_scales;
    d3.out[0]          = r.ws.act;
    d3.dim[0]          = 0;
    d3.dim[1]          = 16;
    d3.dim[3]          = 0;   // routed paths, linear j over 8*512
    d3.dim[6]          = 1;
    d3.slice_count     = kD3RoutedSlices;
    d3.done_counter    = r.alloc_ctr();
    d3.wait_counter[0] = c_d2_ids;
    d3.wait_target[0]  = 1;
    const std::uint32_t c_d3 = d3.done_counter;
    r.push(d3);

    MkInstr d4         = r.blank();
    d4.op              = MkOp::MoeD4;
    d4.ptr[0]          = r.ws.ids;
    d4.ptr[1]          = r.ws.alpha;
    d4.ptr[2]          = r.ws.sscale;
    d4.ptr[3]          = r.ws.act;
    d4.ptr[4]          = a.rd_codes;
    d4.ptr[5]          = a.rd_high;
    d4.ptr[6]          = a.rd_scales;
    d4.ptr[7]          = a.sd_codes;
    d4.out[1]          = const_cast<void*>(a.sd_scales);
    d4.out[0]          = a.x;
    d4.dim[0]          = 0;
    d4.dim[1]          = 16;
    d4.dim[5]          = a.rd_is_q6 ? 1 : 0;
    d4.dim[6]          = 1;
    d4.slice_count     = kD4Slices;
    d4.done_counter    = r.alloc_ctr();
    d4.wait_counter[0] = c_d3;
    d4.wait_target[0]  = kD3RoutedSlices;
    // With the d2 split, c_d3 no longer transitively implies alpha/sscale are
    // written (d3 waits only on ids) — d4 reads them, so wait on full d2 too.
    d4.wait_counter[1] = c_d2;
    d4.wait_target[1]  = 1;
    d4.wait_counter[2] = c_d3s;
    d4.wait_target[2]  = kD3SharedSlices;
    r.push(d4);
    r.prev_d4_ctr = d4.done_counter;
    r.prev_d4_cnt = kD4Slices;
    // MoE consumed the mixer chain; next layer's norm waits d4 + full in-proj.
    r.prev_out_ctr = kMkNone;
}

void mk_flush(cudaStream_t stream) {
    Recorder& r = rec();
    if (r.pending.empty()) { return; }

    const std::size_t count = r.pending.size();
    std::string key(reinterpret_cast<const char*>(r.pending.data()), count * sizeof(MkInstr));
    MkInstr* segment      = nullptr;
    MkStream* seg_streams = nullptr;
    auto hit = r.segment_cache.find(key);
    if (hit != r.segment_cache.end()) {
        // Same bytes already on device (uploaded by an earlier eager round):
        // reuse them — during graph capture this leaves NO memcpy nodes, so
        // replays skip 2 parasitic H2D copies per segment per round.
        segment     = hit->second.tape;
        seg_streams = hit->second.streams;
    } else {
        if (r.tape_used + count > r.tape_capacity) {
            throw std::runtime_error("megakernel: tape arena exhausted");
        }
        segment = r.tape_device + r.tape_used;
        // Persistent host staging + async copy: capture-safe and the source stays
        // valid for the graph's lifetime.
        auto* staged = new std::vector<MkInstr>(r.pending);
        MK_CHECK(cudaMemcpyAsync(segment, staged->data(), count * sizeof(MkInstr),
                                 cudaMemcpyHostToDevice, stream));
        r.tape_used += count;

        if (r.streams_used + static_cast<std::size_t>(r.n_sm) > r.streams_capacity) {
            throw std::runtime_error("megakernel: stream arena exhausted");
        }
        seg_streams        = r.streams_device + r.streams_used;
        auto* host_streams = new std::vector<MkStream>(
            static_cast<std::size_t>(r.n_sm),
            MkStream{segment, static_cast<std::uint32_t>(count)});
        MK_CHECK(cudaMemcpyAsync(seg_streams, host_streams->data(),
                                 host_streams->size() * sizeof(MkStream),
                                 cudaMemcpyHostToDevice, stream));
        r.streams_used += static_cast<std::size_t>(r.n_sm);
        r.segment_cache.emplace(std::move(key), Recorder::SegmentSlot{segment, seg_streams});
    }

    const std::size_t seg_off = static_cast<std::size_t>(segment - r.tape_device);
    unsigned long long* seg_wait = r.prof ? r.wait_ns + seg_off : nullptr;
    unsigned long long* seg_exec = r.prof ? r.exec_ns + seg_off : nullptr;
    unsigned long long* seg_smin = r.prof ? r.span + seg_off : nullptr;
    unsigned long long* seg_smax = r.prof ? r.span + r.tape_capacity + seg_off : nullptr;
    // Launch as a programmatic dependent: the island's last kernel triggers at
    // its head, so this segment's preamble (tape broadcast + first-class weight
    // prefetch) overlaps the island; cudaGridDependencySynchronize inside the
    // interpreter gates data reads on the island's completion.
    static const bool use_pdl = [] {
        const char* env = std::getenv("NINFER_MK_PDL");
        return env == nullptr || env[0] != '0';
    }();
    if (use_pdl) {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{};
        config.gridDim  = dim3(static_cast<unsigned>(r.n_sm), 1, 1);
        config.blockDim = dim3(static_cast<unsigned>(kMkThreads), 1, 1);
        config.stream   = stream;
        config.attrs    = &attribute;
        config.numAttrs = 1;
        MK_CHECK(cudaLaunchKernelEx(&config, mk_interpreter_kernel, seg_streams, r.counters, 1,
                                    seg_wait, seg_exec, seg_smin, seg_smax,
                                    r.prof ? r.seg_stamp : nullptr, r.segments_this_round));
    } else {
        mk_interpreter_kernel<<<r.n_sm, kMkThreads, 0, stream>>>(
            seg_streams, r.counters, 1, seg_wait, seg_exec, seg_smin, seg_smax,
            r.prof ? r.seg_stamp : nullptr, r.segments_this_round);
    }
    MK_CHECK(cudaGetLastError());
    r.pending.clear();
    ++r.segments_this_round;
}

void mk_end_round(cudaStream_t stream) {
    Recorder& r = rec();
    mk_flush(stream);
    if (!r.logged_once) {
        std::fprintf(stderr, "[megakernel] round captured: %d segments, %d classes, %u counters\n",
                     r.segments_this_round, r.classes_this_round, r.ctr_next);
        r.logged_once = true;
    }
    if (r.prof) {
        ++r.rounds_done;
        if (r.rounds_done == 50) {
            MK_CHECK(cudaStreamSynchronize(stream));
            const std::size_t n = r.tape_ops.size();
            std::vector<unsigned long long> wait_h(n), exec_h(n);
            MK_CHECK(cudaMemcpy(wait_h.data(), r.wait_ns, n * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
            MK_CHECK(cudaMemcpy(exec_h.data(), r.exec_ns, n * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
            double wait_sum[32] = {};
            double exec_sum[32] = {};
            for (std::size_t i = 0; i < n; ++i) {
                const int op = static_cast<int>(r.tape_ops[i]) & 31;
                wait_sum[op] += static_cast<double>(wait_h[i]);
                exec_sum[op] += static_cast<double>(exec_h[i]);
            }
            std::fprintf(stderr, "[megakernel prof] per-op CTA-clock sums over %d rounds:\n",
                         r.rounds_done);
            for (int op = 0; op < 32; ++op) {
                if (wait_sum[op] + exec_sum[op] == 0.0) { continue; }
                std::fprintf(stderr, "  op%-2d wait %12.0f exec %12.0f (us@2.4GHz)\n", op,
                             wait_sum[op] / 2400.0, exec_sum[op] / 2400.0);
            }
        }
    }
}

} // namespace ninfer::ops::mk
