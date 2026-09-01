#!/usr/bin/env python3
"""Пакет 09 на актуальном мастере: разрешение трёх конфликтов.

Апстримный `70434721` и `a58a946c` переписали ту же область: появились `state_source_slots` и
`state_destination_slots` вместо `lanes`, счётчики токенов в жадной ветке сэмплера, NVTX-обёртки
и переименование `GqaExecutionEnvelope` в `CausalAttentionExecutionEnvelope`. Механизмы разные,
поэтому везде берётся обе стороны, а не одна.
"""
import io
import os
import re
import sys

R = os.environ.get('NINFER_TREE', '/root/ninfer_d4')

CONFLICT = re.compile(r'<<<<<<< [^\n]*\n(.*?)\n?=======\n(.*?)>>>>>>> [^\n]*\n', re.S)


def resolve(path, resolutions):
    full = os.path.join(R, path)
    s = io.open(full, encoding='utf-8').read()
    blocks = list(CONFLICT.finditer(s))
    if len(blocks) != len(resolutions):
        sys.exit('%s: гнёзд %d, решений %d' % (path, len(blocks), len(resolutions)))
    out = []
    last = 0
    for m, text in zip(blocks, resolutions):
        out.append(s[last:m.start()])
        out.append(text)
        last = m.end()
    out.append(s[last:])
    io.open(full, 'w', encoding='utf-8', newline='\n').write(''.join(out))
    print('разрешён ' + path)


# --- 1. сэмплер: обе стороны ---------------------------------------------------------------
resolve('src/ops/kernel/sampling.cuh', [
    """            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[red_idx[0]], 1); }
            // A greedy row is a one-point proposal, so its probability is one.
            if (out_prob != nullptr) { out_prob[row] = 1.0f; }
            if (out_sup_idx != nullptr) {
                for (int j = 0; j < sup_cap; ++j) {
                    out_sup_idx[row * sup_cap + j]  = j == 0 ? red_idx[0] : 0;
                    out_sup_prob[row * sup_cap + j] = j == 0 ? 1.0f : 0.0f;
                }
                out_sup_n[row] = 1;
            }
""",
    """            const int picked = sampling_key_index(best);
            out[col]         = picked;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
            // A greedy row is a one-point proposal, so its probability is one.
            if (out_prob != nullptr) { out_prob[col] = 1.0f; }
            if (out_sup_idx != nullptr) {
                for (int j = 0; j < sup_cap; ++j) {
                    out_sup_idx[col * sup_cap + j]  = j == 0 ? picked : 0;
                    out_sup_prob[col * sup_cap + j] = j == 0 ? 1.0f : 0.0f;
                }
                out_sup_n[col] = 1;
            }
""",
])

# --- 2. объявление: тип апстрима, параметры наши --------------------------------------------
resolve('src/targets/qwen3_6/impl/runtime/text_context.h', [
    """                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens,
                           Tensor* draft_probs = nullptr, const Tensor* positions = nullptr,
                           std::int32_t purpose_offset         = 0,
                           const ops::SamplingConfig* sampling = nullptr,
                           Tensor* support_ids = nullptr, Tensor* support_probs = nullptr,
                           Tensor* support_n = nullptr);
""",
])

# --- 3. mtp_impl: три гнезда ------------------------------------------------------------------
SLICES = """        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows           = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents       = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids      = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden   = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden          = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden        = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions       = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions  = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns   = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts        = frame.next_drafts.slice(0, 0, batch_size);
        // Acceptance takes the unsliced tensors: their row stride comes from the first dimension,
        // and there can be fewer active rows than the concurrency maximum.
        Tensor draft_probs        = frame.draft_probs;
        Tensor draft_sup_ids      = frame.draft_support_ids;
        Tensor draft_sup_probs    = frame.draft_support_probs;
        Tensor draft_sup_n        = frame.draft_support_n;
        Tensor draft_recorded     = frame.draft_recorded_tokens;
        // Write slice: the step first, then the prefix of active rows, so it stays contiguous.
        auto step_rows = [batch_size](const Tensor& t, std::int32_t step, std::int32_t width) {
            return t.slice(1, step, 1).slice(0, 0, width).view({width});
        };
"""

VERIFY = """        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_rope,
                                     .valid_columns           = target_valid,
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = current_drafts,
                                     .current_extents         = current_extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .selected_hidden         = selected_hidden,
                                     .draft_probs             = draft_probs,
                                     .draft_support_ids       = draft_sup_ids,
                                     .draft_support_probs     = draft_sup_probs,
                                     .draft_support_n         = draft_sup_n,
                                     .draft_recorded_tokens   = draft_recorded,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                 },
                                 envelopes.target_verify);
        }
"""

PROPOSE = """            Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
            Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
            Tensor prob0           = step_rows(draft_probs, 0, batch_size);
            Tensor sup_ids0        = step_rows(draft_sup_ids, 0, 20 * batch_size);
            Tensor sup_probs0      = step_rows(draft_sup_probs, 0, 20 * batch_size);
            Tensor sup_n0          = step_rows(draft_sup_n, 0, batch_size);
            card.mtp_propose_batch(ar_hidden, proposal_logits, draft0, &prob0, &frontiers, 0,
                                   frame.sampling, &sup_ids0, &sup_probs0, &sup_n0);
            Tensor rec0 = step_rows(draft_recorded, 0, batch_size);
            CUDA_CHECK(cudaMemcpyAsync(rec0.data, draft0.data, rec0.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next =
                    next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
                Tensor position =
                    ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
                Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
                Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              mtp_rows, envelopes.ar[step], next_hidden_batch);
                const auto s1     = static_cast<std::int32_t>(step + 1);
                Tensor next_prob  = step_rows(draft_probs, s1, batch_size);
                Tensor next_sup   = step_rows(draft_sup_ids, s1, 20 * batch_size);
                Tensor next_supp  = step_rows(draft_sup_probs, s1, 20 * batch_size);
                Tensor next_supn  = step_rows(draft_sup_n, s1, batch_size);
                card.mtp_propose_batch(next_hidden, proposal_logits, next, &next_prob, &frontiers,
                                       s1, frame.sampling, &next_sup, &next_supp, &next_supn);
                Tensor next_rec = step_rows(draft_recorded, s1, batch_size);
                CUDA_CHECK(cudaMemcpyAsync(next_rec.data, next.data, next_rec.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
            }
"""

resolve('src/targets/qwen3_6/impl/runtime/mtp_impl.h', [SLICES, VERIFY, PROPOSE])
print('все гнёзда разрешены')
