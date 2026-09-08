#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_pack_fc_input
 *
 * Math / indexing:
 *   out[0:D, t] = embedding_norm[:, t]
 *   out[D:2D, t] = hidden_norm[:, t]
 *
 * Logical shapes:
 *   BF16 embedding_norm and hidden_norm [D,T], out [2D,T], contiguous. The registered domains are
 *   D=5120 for Qwen3.6-27B and D=2048 for Qwen3.6-35B-A3B.
 *
 * Numeric:
 *   Exact BF16 element copies; no arithmetic or conversion.
 *
 * Effects:
 *   Writes the full output. Inputs and output must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream);

/**
 * Op: mtp_norm_pack_fc_input
 *
 * Math / indexing:
 *   out[0:D, t]  = rmsnorm(embedding[:, t], embedding_weight)
 *   out[D:2D, t] = rmsnorm(hidden[:, t], hidden_weight)
 *   with the unit-offset weight convention, i.e. the weight used is (1 + w).
 *
 * Logical shapes:
 *   BF16 embedding and hidden [D,T], weights [D], out [2D,T], all contiguous.
 *
 * Numeric:
 *   Identical to rmsnorm() applied to each half followed by mtp_pack_fc_input(): the same block
 *   width, pair decomposition, reduction and epilogue produce the same bytes. Returns false from
 *   mtp_norm_pack_fc_input_supported() for shapes it does not cover, and the caller keeps the
 *   three-Op path there.
 *
 * Effects:
 *   Writes the full output. Inputs, weights and output must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
[[nodiscard]] bool mtp_norm_pack_fc_input_supported(const Tensor& embedding,
                                                    const Tensor& embedding_weight,
                                                    const Tensor& hidden,
                                                    const Tensor& hidden_weight, const Tensor& out);

void mtp_norm_pack_fc_input(const Tensor& embedding, const Tensor& embedding_weight,
                            const Tensor& hidden, const Tensor& hidden_weight, Tensor& out,
                            float eps, cudaStream_t stream);

/**
 * Op: mtp_residual_norm
 *
 * Math / indexing:
 *   residual[:, t] += delta[:, t]           (pairwise, FP32 sum, round to nearest BF16)
 *   out[:, t]       = rmsnorm(residual[:, t], weight)   with the unit-offset weight convention.
 *
 * Logical shapes:
 *   BF16 delta and residual [D,T], weight [D], out [D,T], all contiguous.
 *
 * Numeric:
 *   Identical to residual_add() followed by rmsnorm() on the updated residual: the same pairwise
 *   FP32 addition and rounding, then the same block width, pair decomposition, reduction and
 *   epilogue. mtp_residual_norm_supported() reports the shapes this route covers; the caller keeps
 *   the two-Op path elsewhere.
 *
 * Effects:
 *   Updates the full residual in place and writes the full output. delta, weight and out must not
 *   alias the residual.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
[[nodiscard]] bool mtp_residual_norm_supported(const Tensor& delta, const Tensor& residual,
                                               const Tensor& weight, const Tensor& out);

void mtp_residual_norm(const Tensor& delta, Tensor& residual, const Tensor& weight, Tensor& out,
                       float eps, cudaStream_t stream);

/**
 * Op: mtp_split_attn_in
 *
 * Math / indexing:
 *   For each token, rows [0,6144), [6144,7168), [7168,13312), and [13312,14336) are copied to
 *   flattened Q[6144], K[1024], Gate[6144], and V[1024], respectively.
 *
 * Logical shapes:
 *   attn_in [14336,T]; q/gate [256,24,T]; k/v [256,4,T], all contiguous BF16.
 *
 * Numeric:
 *   Exact BF16 element copies with only an index remap.
 *
 * Effects:
 *   Writes every output element. Outputs and input must be pairwise non-aliasing.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream);

} // namespace ninfer::ops
