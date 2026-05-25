#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <util/metal.hpp>

#include "Metal/MTLCommandBuffer.hpp"

namespace tensor::metal {

class AutoreleasePool {
public:
  AutoreleasePool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
  ~AutoreleasePool() {
    pool_->release();
  }

  AutoreleasePool(const AutoreleasePool&) = delete;
  AutoreleasePool& operator=(const AutoreleasePool&) = delete;

private:
  NS::AutoreleasePool* pool_;
};

class MetalContext {
public:
  static MetalContext& instance();

  [[nodiscard]] NS::SharedPtr<MTL::Device> device() const {
    return device_;
  }
  [[nodiscard]] NS::SharedPtr<MTL::CommandQueue> queue() const {
    return queue_;
  }
  [[nodiscard]] NS::SharedPtr<MTL::Library> library() const {
    return library_;
  }

  NS::SharedPtr<MTL::ComputePipelineState> pso(std::string_view fn_name);
  void track(NS::SharedPtr<MTL::CommandBuffer> last_cmd);
  void synchronize();

private:
  MetalContext();
  ~MetalContext();

  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::CommandBuffer> last_cmd_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> pso_cache_;
};

} // namespace tensor::metal
