#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "core/l2_persist.h"
#include "core/nvtx.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch(state.execution.device.stream);
    } else {
        nvtx::ScopedRange eager_range(nvtx::Name::DecodeEager, nvtx::Category::Decode);
        body();
    }
}

template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    // Keep the Linear Attention continuation state resident in L2 for the captured decode
    // graph. Every decode round reads the whole pool and writes it back at a fixed address,
    // and recurrent_fold_kernel streams all of it in a single launch, so it is the one decode
    // consumer with both a fixed footprint and enough reuse to be worth an L2 reservation.
    // The window must be installed before capture: the graph bakes it into its kernel nodes
    // and does not re-read the stream attribute at replay.
    const LinearAttentionStateAllLayersView linear =
        state.execution.linear_attention.all_layers_view();
    const auto low = reinterpret_cast<std::uintptr_t>(linear.conv_layer0.data);
    const auto span =
        static_cast<std::size_t>(static_cast<std::uintptr_t>(linear.recurrent_layer_stride_bytes) *
                                     (linear.spec.layers - 1U) +
                                 reinterpret_cast<std::uintptr_t>(linear.recurrent_layer0.data) -
                                 low + linear.recurrent_layer0.bytes());
    // A target whose state pool does not fit an access policy window gets nothing installed,
    // and then there is nothing to take off the stream either.
    const bool pinned =
        l2p::pin_range(state.execution.device.stream, reinterpret_cast<const void*>(low), span);
    definition.capture(state.execution.device.stream, body);
    if (pinned) { l2p::unpin(state.execution.device.stream); }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
