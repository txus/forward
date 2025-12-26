#include <cuda_runtime.h>
#include <tensor/storage.hpp>
#include <tensor/device_type.hpp>
#include "kernels/fill.cuh"
#include "kernels/utils.cuh"

namespace tensor {

using namespace dtype;
using namespace device;
using namespace kernels;

template <typename T>
TensorStorage<T, CUDA>::TensorStorage(size_t size) : size_(size) {
  if (size > 0) {
    size_t padded_size = (size + 7) & ~7; // Round up to multiple of 8 for vectorized kernels
    cudaMalloc(&data_, padded_size * sizeof(T));
  }
}

template <typename T>
TensorStorage<T, CUDA>::~TensorStorage() {
  if (data_) {
    cudaFree(data_);
  }
}

template <typename T>
TensorStorage<T, CUDA>::TensorStorage(TensorStorage&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

template <typename T>
void TensorStorage<T, CUDA>::resize(size_t size) {
  if (data_) { cudaFree(data_); }
  size_ = size;
  if (size > 0) {
    size_t padded_size = (size + 7) & ~7; // Round up to multiple of 8 for vectorized kernels
    cudaMalloc(&data_, padded_size * sizeof(T));
  }
}

template <typename T>
void TensorStorage<T, CUDA>::fill(T value) {
  if (size_ == 0) { return; }

  int block_size = cuda::get_block_size();
  int grid_size = cuda::get_grid_size(size_, block_size);

  // Convert to device-native type for kernel call
  auto* device_data = reinterpret_cast<Cuda<T>*>(data_); // NOLINT
  Cuda<T> device_value = to_device_type(value, CUDA{});

  fill_kernel<<<grid_size, block_size>>>(device_data, device_value, size_);
  CUDA_CHECK(cudaGetLastError()); // NOLINT
}

// Const CUDA storage implementations
template <typename T>
TensorStorage<const T, CUDA>::TensorStorage(size_t size) : size_(size) {
  if (size > 0) {
    size_t padded_size = (size + 7) & ~7; // Round up to multiple of 8 for vectorized kernels
    cudaMalloc(&data_, padded_size * sizeof(T));
  }
}

template <typename T>
TensorStorage<const T, CUDA>::~TensorStorage() {
  if (data_) {
    cudaFree(data_);
  }
}

template <typename T>
TensorStorage<const T, CUDA>::TensorStorage(TensorStorage&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

template <typename T>
TensorStorage<const T, CUDA>& TensorStorage<const T, CUDA>::operator=(TensorStorage&& other) noexcept {
  if (this != &other) {
    if (data_) { cudaFree(data_); }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

template <typename T>
void TensorStorage<const T, CUDA>::resize(size_t size) {
  if (data_) { cudaFree(data_); }
  size_ = size;
  if (size > 0) {
    cudaMalloc(&data_, size * sizeof(T));
  }
}

// Explicit instantiations
template class TensorStorage<float, CUDA>;
template class TensorStorage<bfloat16, CUDA>;
template class TensorStorage<int, CUDA>;

template class TensorStorage<const float, CUDA>;
template class TensorStorage<const bfloat16, CUDA>;
template class TensorStorage<const int, CUDA>;
};
