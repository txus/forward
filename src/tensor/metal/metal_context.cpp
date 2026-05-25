#include "metal_context.hpp"

#include <fmt/format.h>

namespace tensor::metal {

MetalContext::MetalContext()
    : device_(NS::TransferPtr(MTL::CreateSystemDefaultDevice())),
      queue_(NS::TransferPtr(device_->newCommandQueue())) {
  NS::Error* error = nullptr;
  NS::String* filePath = NS::String::string(METAL_DEFAULT_LIBRARY_PATH, NS::UTF8StringEncoding);
  library_ = NS::TransferPtr(device_->newLibrary(filePath, &error));
}

MetalContext& MetalContext::instance() {
  static MetalContext ctx;
  return ctx;
}

MetalContext::~MetalContext() = default;

void MetalContext::track(NS::SharedPtr<MTL::CommandBuffer> last_cmd) {
  last_cmd_ = last_cmd;
}

void MetalContext::synchronize() {
  if (last_cmd_) {
    last_cmd_->waitUntilCompleted();
    last_cmd_ = nullptr;
  }
}

NS::SharedPtr<MTL::ComputePipelineState> MetalContext::pso(std::string_view fn_name) {
  const std::string function_name(fn_name);
  auto iterator = pso_cache_.find(function_name);
  if (iterator != pso_cache_.end()) {
    return iterator->second;
  }

  NS::Error* error = nullptr;
  auto function =
      library_->newFunction(NS::String::string(function_name.c_str(), NS::UTF8StringEncoding));
  if (!function) {
    fmt::println("Could not find function {} in kernels library", fn_name);
    exit(1);
  }
  auto pso = NS::TransferPtr(device_->newComputePipelineState(function, &error));
  if (error != nullptr) {
    fmt::println("Could not create new compute pipeline state for function {}", fn_name);
    exit(1);
  }
  pso_cache_[function_name] = pso;
  return pso;
}

} // namespace tensor::metal
