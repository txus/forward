#include <tensor/storage.hpp>

#include "launcher.hpp"
#include "metal_context.hpp"

namespace tensor {
using namespace dtype;
using namespace device;
using namespace metal;

struct metal_fwd::BufferHandle {
  MTL::Buffer* buf;
};

void metal_fwd::synchronize() {
  metal::MetalContext::instance().synchronize();
}

template <typename T> TensorStorage<T, METAL>::TensorStorage(size_t size) : size_(size) {
  if (size > 0) {
    auto device = MetalContext::instance().device();
    size_t padded = (size + 7) & ~7;
    auto* buf = device->newBuffer(padded * sizeof(T), MTL::ResourceStorageModeShared);
    buffer_ =
        std::shared_ptr<metal_fwd::BufferHandle>(new metal_fwd::BufferHandle(buf), [](auto* h) {
          h->buf->release();
          delete h;
        });
    data_ = reinterpret_cast<T*>(buf->contents());
  }
}

template <typename T> void TensorStorage<T, METAL>::fill(T value) {
  if (size_ == 0) {
    return;
  }
  if constexpr (std::is_same_v<T, float>) {
    uint32_t n = size_;
    metal::launch("fill_f32", size_,
                  {Buffer{.buf = buffer_->buf}, Bytes{.data = &value, .size = sizeof(value)},
                   Bytes{.data = &n, .size = sizeof(n)}});
  }
}

template class TensorStorage<float, METAL>;
template class TensorStorage<bfloat16, METAL>;
template class TensorStorage<int, METAL>;
template class TensorStorage<const float, METAL>;
template class TensorStorage<const bfloat16, METAL>;
template class TensorStorage<const int, METAL>;

} // namespace tensor
