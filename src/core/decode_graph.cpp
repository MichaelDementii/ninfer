#include "core/decode_graph.h"

#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <cstddef>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

std::size_t g_last_instantiated_nodes = 0;

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    {
        std::size_t captured = 0;
        if (cudaGraphGetNodes(definition.graph_, nullptr, &captured) == cudaSuccess) {
            g_last_instantiated_nodes = captured;
        }
    }
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        CUDA_CHECK(err);
    }
    exec_ = exec;
}

namespace {



std::string graph_update_shape_description(cudaGraph_t replacement) {
    std::size_t now = 0;
    if (cudaGraphGetNodes(replacement, nullptr, &now) != cudaSuccess) { return ""; }
    return "; captured " + std::to_string(g_last_instantiated_nodes) + " nodes, now " +
           std::to_string(now);
}

// A topology change names a node; report which one, and for a kernel node which function, so the
// message points at the route that moved rather than at the round as a whole.
std::string graph_update_node_description(const cudaGraphExecUpdateResultInfo& info) {
    const cudaGraphNode_t node = info.errorNode != nullptr ? info.errorNode : info.errorFromNode;
    if (node == nullptr) { return " at an unnamed node"; }

    cudaGraphNodeType type{};
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess) { return " at an unreadable node"; }
    if (type != cudaGraphNodeTypeKernel) {
        return " at a non-kernel node of type " + std::to_string(static_cast<int>(type));
    }

    cudaKernelNodeParams params{};
    if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
        return " at a kernel node whose parameters could not be read";
    }
    const char* name = nullptr;
    if (cudaFuncGetName(&name, params.func) != cudaSuccess || name == nullptr) {
        return " at an unnamed kernel node";
    }
    return std::string(" at kernel ") + name + " grid " + std::to_string(params.gridDim.x) + "x" +
           std::to_string(params.gridDim.y) + " block " + std::to_string(params.blockDim.x);
}

} // namespace

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")" +
            graph_update_node_description(result) +
            graph_update_shape_description(definition.graph_));
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphUpload(exec_, stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer
