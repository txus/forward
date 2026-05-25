#include <stdexcept>
#include <tensor/storage.hpp>
#include <type_traits>

#include "buffer_handle.hpp"
#include "launcher.hpp"
#include "metal_context.hpp"

namespace tensor {
using namespace dtype;
using namespace device;
using namespace metal;

void metal_fwd::synchronize() {
  metal::MetalContext::instance().synchronize();
}

template <typename T> TensorStorage<T, METAL>::TensorStorage(size_t size) : size_(size) {
  resize(size);
}

template <typename T> void TensorStorage<T, METAL>::fill(T value) {
  if (size_ == 0) {
    return;
  }
  const char* fn_name = nullptr;
  if constexpr (std::is_same_v<std::remove_const_t<T>, bfloat16>) {
    fn_name = "fill_bf16";
  } else if constexpr (std::is_same_v<std::remove_const_t<T>, float>) {
    fn_name = "fill_fp32";
  } else if constexpr (std::is_same_v<std::remove_const_t<T>, int>) {
    fn_name = "int_i32";
  } else {
    throw std::runtime_error("Fill only supports fp32, int32 and bf16 on Metal");
  }
  uint32_t n = size_;
  metal::launch(fn_name, size_,
                {Buffer{.buf = buffer_->buf}, Bytes{.data = &value, .size = sizeof(value)},
                 Bytes{.data = &n, .size = sizeof(n)}});
}

template <typename T> void TensorStorage<T, METAL>::resize(size_t size) {
  size_ = size;
  if (size == 0) {
    buffer_.reset();
    data_ = nullptr;
    return;
  }
  auto device = MetalContext::instance().device();
  size_t padded = (size + 7) & ~7;
  auto* buf = device->newBuffer(padded * sizeof(T), MTL::ResourceStorageModeShared);
  buffer_ = std::shared_ptr<metal_fwd::BufferHandle>(new metal_fwd::BufferHandle(buf), [](auto* h) {
    h->buf->release();
    delete h;
  }); // assigning here releases the previous buffer_
  data_ = reinterpret_cast<T*>(buf->contents());
}

template class TensorStorage<float, METAL>;
template class TensorStorage<bfloat16, METAL>;
template class TensorStorage<int, METAL>;
template class TensorStorage<const float, METAL>;
template class TensorStorage<const bfloat16, METAL>;
template class TensorStorage<const int, METAL>;

} // namespace tensor
