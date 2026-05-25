#pragma once
#include <string_view>
#include <tensor/device.hpp>
#include <tensor/tensor.hpp>
#include <variant>

#include "buffer_handle.hpp" // complete BufferHandle — needed by to_buffer (used via templates)
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

// A whole Metal tensor binds its buffer at offset 0.
template <typename T> inline Buffer to_buffer(const Tensor<T, METAL>& tensor) {
  return Buffer{.buf = tensor.mtl_handle()->buf};
}

// A view binds its owning buffer at the byte offset where the view begins.
// `setBuffer(buf, offset, idx)` makes the kernel's `device T*` start there, so
// the kernel is oblivious to slicing — exactly like CUDA's `base + offset`.
template <typename T> inline Buffer to_buffer(const TensorView<T, METAL>& view) {
  auto* base = reinterpret_cast<const std::byte*>(view.buf->buf->contents());
  auto* here = reinterpret_cast<const std::byte*>(view.data);
  return Buffer{.buf = view.buf->buf, .offset = static_cast<size_t>(here - base)};
}

// Upload a scalar inline (setBytes copies it during encoding, so `value` only
// needs to outlive the launch call). Pass the value itself, not its address.
template <typename T> inline Bytes to_bytes(const T& value) {
  return Bytes{.data = &value, .size = sizeof(T)};
}

inline void launch(const std::string_view fn_name, size_t grid_x,
                   std::initializer_list<std::variant<Buffer, Bytes>> args) {
  AutoreleasePool pool;
  auto& ctx = MetalContext::instance();
  auto pso = ctx.pso(fn_name);
  auto cmd = NS::RetainPtr(ctx.queue()->commandBuffer());
  auto* enc = cmd->computeCommandEncoder();

  enc->setComputePipelineState(pso.get());
  uint32_t idx = 0;
  for (const auto& arg : args) {
    std::visit(
        [&](auto&& argument) {
          using ArgT = std::decay_t<decltype(argument)>;
          if constexpr (std::is_same_v<ArgT, Buffer>) {
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
