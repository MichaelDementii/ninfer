#pragma once

// Megakernel engine integration (v0.4a): records GDN-mixer and MoE task classes
// during decode graph capture (T=1, batch=1) and launches persistent interpreter
// segments in place of the per-op kernels. Attention mixers stay native; a
// segment flush happens before each one, so stream order preserves the round's
// dataflow. Enabled with NINFER_MEGAKERNEL=1.
//
// All recorded pointers are the exact arguments the ops would have received;
// intermediates live in a recorder-owned workspace so the tape stays valid for
// the lifetime of the captured graph.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::mk {

bool mk_engine_enabled();

struct MkGdnMixerArgs {
    // residual / hidden
    void* x;                       // bf16[2048], read by norm, residual target of out-proj
    // input norm
    const void* input_norm_w;      // bf16[2048]
    float rms_eps;
    // gating (a/b projections, 32x2048 bf16 each) + transform
    const void* a_w;               // bf16[32*2048]
    const void* b_w;               // bf16[32*2048]
    const void* a_log;             // f32[32]
    const void* dt_bias;           // f32[32]
    // in-projection (W8 12288x2048) + conv
    const void* qkvz_codes;        // s8
    const void* qkvz_scales;       // u16
    const void* conv_w;            // bf16[4*8192] plane-wise
    void* conv_state;              // bf16[3*8192 * slots]
    const void* state_slots;       // i32[1] device (initial == snapshot for decode)
    // recurrent
    void* rec_state;               // f32, slot-strided
    std::int64_t rec_slot_stride;  // 128*128*ne2
    float gdn_scale;
    // gated norm + out-projection (W8 2048x4096)
    const void* gdn_norm_w;        // bf16[128]
    const void* out_codes;
    const void* out_scales;
};

struct MkMoeArgs {
    void* x;                    // bf16[2048] residual (norm2 input + d4 destination)
    const void* post_norm_w;    // bf16[2048]
    float rms_eps;
    const void* router;         // bf16[257*2048]
    const void* rgu_codes;      // routed gate_up Q4
    const void* rgu_scales;
    const void* rd_codes;       // routed down Q5/Q6
    const void* rd_high;
    const void* rd_scales;
    int rd_is_q6;               // layers 34/38/39 use Q6G64
    const void* sgu_codes;      // shared gate_up W8
    const void* sgu_scales;
    const void* sd_codes;       // shared down W8
    const void* sd_scales;
};

struct MkAttnArgs {
    void* x;                       // bf16[2048] residual
    const void* input_norm_w;      // bf16[2048]
    float rms_eps;
    const void* qkgv_codes;        // W8 9216x2048
    const void* qkgv_scales;
    void* q;                       // engine workspace views (graph-static)
    void* gate;                    // bf16[4096]
    void* k;                       // bf16[512]
    void* v;                       // bf16[512]
    const void* q_norm_w;          // bf16[256]
    const void* k_norm_w;          // bf16[256]
    void* qn;                      // bf16[4096]
    void* kn;                      // bf16[512]
    const void* rope_positions;    // i32 device
    void* attn;                    // bf16[4096] (gqa output a)
    const void* o_codes;           // W8 2048x4096
    const void* o_scales;
};

// Round lifecycle (called during graph capture only).
void mk_begin_round(cudaStream_t stream);
void mk_record_gdn_mixer(const MkGdnMixerArgs& args);
void mk_record_moe(const MkMoeArgs& args);
// Attention absorption: pre-classes (norm, qkv split, q/k norms, rope), then the
// caller flushes and runs the native gqa partial+reduce, then post-classes
// (sigmoid gate mul, o-proj residual).
void mk_record_attn_pre(const MkAttnArgs& args);
void mk_record_attn_post(const MkAttnArgs& args);
bool mk_attn_enabled();
void mk_flush(cudaStream_t stream);      // launch interpreter for pending classes
void mk_end_round(cudaStream_t stream);  // final flush + stats

} // namespace ninfer::ops::mk
