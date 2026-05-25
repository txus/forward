#pragma once
#include <string_view>
#include <variant>

#include "Foundation/NSSharedPtr.hpp"
#include "metal_context.hpp"

namespace tensor::metal {
struct Buffer {
  MTL::Buffer* buf;
  size_t offset = 0;
};
struct Bytes {
  const void* data;
  size_t size;
};

inline void launch(const std::string_view fn_name, size_t grid_x,
                   std::initializer_list<std::variant<Buffer, Bytes>> args) {
  AutoreleasePool pool;
  auto& ctx = MetalContext::instance();
  auto pso = ctx.pso(fn_name);
  auto cmd = NS::RetainPtr(ctx.queue()->commandBuffer());
  auto* enc = cmd->computeCommandEncoder();

  enc->setComputePipelineState(pso);
  uint32_t idx = 0;
  for (const auto& arg : args) {
    std::visit(
        [&](auto&& argument) {
          using T = std::decay_t<decltype(argument)>;
          if constexpr (std::is_same_v<T, Buffer>) {
            enc->setBuffer(argument.buf, argument.offset, idx);
          } else {
            enc->setBytes(argument.data, argument.size, idx);
          }
          ++idx;
        },
        arg);
  }
  MTL::Size grid(grid_x, 1, 1);
  MTL::Size tgs(std::min<NS::UInteger>(pso->maxTotalThreadsPerThreadgroup(), 256), 1, 1);
  enc->dispatchThreads(grid, tgs);
  enc->endEncoding();
  cmd->commit();
  ctx.track(cmd);
}

} // namespace tensor::metal
