#include "ops/megakernel/mk_engine.h"

#include "ops/megakernel/mk_core.cuh"
#include "ops/megakernel/mk_instr.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::mk {
namespace {

constexpr int kRowSlice   = 16;
constexpr int kInSlices   = 12288 / kRowSlice;   // 768
constexpr int kHeadSlices = 8192 / kRowSlice;    // 512
constexpr int kOutSlices  = 2048 / kRowSlice;    // 128
constexpr int kRecSlices  = (32 * 32) / 16;      // 64
constexpr int kD1Slices   = 129;                 // 2 router rows per slice
constexpr int kD3Slices   = 512 / 4;             // 128
constexpr int kD4Slices   = 2048 / 8;            // 256

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
};

struct Recorder {
    bool ready          = false;
    bool prof           = false;
    int rounds_done     = 0;
    unsigned long long* wait_ns = nullptr;
    unsigned long long* exec_ns = nullptr;
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
        if (rounds_done == 0) { tape_ops.push_back(instr.op); }
        ++classes_this_round;
    }
};

Recorder& rec() {
    static Recorder r;
    return r;
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
}

void mk_record_gdn_mixer(const MkGdnMixerArgs& a) {
    Recorder& r = rec();

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
    const std::uint32_t c_norm = norm.done_counter;
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
    const std::uint32_t c_ga = ga.done_counter;
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
    const std::uint32_t c_gb = gb.done_counter;
    r.push(gb);

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
    in.dim[1]          = kRowSlice;
    in.slice_count     = kInSlices;
    in.done_counter    = r.alloc_ctr();
    in.done2_counter   = r.alloc_ctr();
    in.done2_limit     = kHeadSlices;
    in.wait_counter[0] = c_norm;
    in.wait_target[0]  = 1;
    const std::uint32_t c_in      = in.done_counter;
    const std::uint32_t c_in_head = in.done2_counter;
    r.push(in);
    r.prev_in_ctr = c_in;
    r.prev_in_cnt = kInSlices;

    MkInstr gg         = r.blank();
    gg.op              = MkOp::GdnGating;
    gg.ptr[0]          = r.ws.ab;
    gg.ptr[1]          = r.ws.ab + 32;
    gg.ptr[2]          = a.a_log;
    gg.ptr[3]          = a.dt_bias;
    gg.out[0]          = r.ws.gf;
    gg.out[1]          = r.ws.betaf;
    gg.dim[0]          = 32;
    gg.done_counter    = r.alloc_ctr();
    gg.wait_counter[0] = c_ga;
    gg.wait_target[0]  = 1;
    gg.wait_counter[1] = c_gb;
    gg.wait_target[1]  = 1;
    const std::uint32_t c_gg = gg.done_counter;
    r.push(gg);

    MkInstr rc         = r.blank();
    rc.op              = MkOp::GdnRecurrent;
    rc.ptr[0]          = r.ws.qc;
    rc.ptr[1]          = r.ws.kc;
    rc.ptr[2]          = r.ws.vc;
    rc.ptr[3]          = r.ws.gf;
    rc.ptr[4]          = r.ws.betaf;
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
    rc.wait_target[0]  = kHeadSlices;
    rc.wait_counter[1] = c_gg;
    rc.wait_target[1]  = 1;
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
    gn.wait_target[1]  = kInSlices;
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

    MkInstr d2         = r.blank();
    d2.op              = MkOp::MoeD2;
    d2.ptr[0]          = r.ws.scores;
    d2.ptr[1]          = r.ws.sscale;
    d2.out[0]          = r.ws.ids;
    d2.out[1]          = r.ws.alpha;
    d2.done_counter    = r.alloc_ctr();
    d2.wait_counter[0] = c_d1;
    d2.wait_target[0]  = kD1Slices;
    const std::uint32_t c_d2 = d2.done_counter;
    r.push(d2);

    MkInstr d3         = r.blank();
    d3.op              = MkOp::MoeD3;
    d3.ptr[0]          = r.ws.h2;
    d3.ptr[1]          = r.ws.ids;
    d3.ptr[2]          = a.rgu_codes;
    d3.ptr[4]          = a.rgu_scales;
    d3.ptr[5]          = a.sgu_codes;
    d3.ptr[6]          = a.sgu_scales;
    d3.out[0]          = r.ws.act;
    d3.dim[0]          = 0;
    d3.dim[1]          = 4;
    d3.dim[6]          = 1;
    d3.slice_count     = kD3Slices;
    d3.done_counter    = r.alloc_ctr();
    d3.wait_counter[0] = c_d2;
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
    d4.dim[1]          = 8;
    d4.dim[5]          = a.rd_is_q6 ? 1 : 0;
    d4.dim[6]          = 1;
    d4.slice_count     = kD4Slices;
    d4.done_counter    = r.alloc_ctr();
    d4.wait_counter[0] = c_d3;
    d4.wait_target[0]  = kD3Slices;
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
    if (r.tape_used + count > r.tape_capacity) {
        throw std::runtime_error("megakernel: tape arena exhausted");
    }
    MkInstr* segment = r.tape_device + r.tape_used;
    // Persistent host staging + async copy: capture-safe (becomes a graph memcpy
    // node) and the source stays valid for the graph's lifetime.
    auto* staged = new std::vector<MkInstr>(r.pending);
    MK_CHECK(cudaMemcpyAsync(segment, staged->data(), count * sizeof(MkInstr),
                             cudaMemcpyHostToDevice, stream));
    r.tape_used += count;

    if (r.streams_used + static_cast<std::size_t>(r.n_sm) > r.streams_capacity) {
        throw std::runtime_error("megakernel: stream arena exhausted");
    }
    MkStream* seg_streams = r.streams_device + r.streams_used;
    auto* host_streams    = new std::vector<MkStream>(
        static_cast<std::size_t>(r.n_sm),
        MkStream{segment, static_cast<std::uint32_t>(count)});
    MK_CHECK(cudaMemcpyAsync(seg_streams, host_streams->data(),
                             host_streams->size() * sizeof(MkStream), cudaMemcpyHostToDevice,
                             stream));
    r.streams_used += static_cast<std::size_t>(r.n_sm);

    unsigned long long* seg_wait =
        r.prof ? r.wait_ns + static_cast<std::size_t>(segment - r.tape_device) : nullptr;
    unsigned long long* seg_exec =
        r.prof ? r.exec_ns + static_cast<std::size_t>(segment - r.tape_device) : nullptr;
    mk_interpreter_kernel<<<r.n_sm, kMkThreads, 0, stream>>>(seg_streams, r.counters, 1, seg_wait,
                                                             seg_exec);
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
