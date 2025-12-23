#include <cuda_runtime.h>
#include <tensor/storage.hpp>
#include <tensor/dtype.hpp>
#include "kernels/fill.cuh"
#include "utils.cuh"

namespace tensor {

using namespace dtype;
using namespace device;

template <DType T>
TensorStorage<T, CUDA>::TensorStorage(int size) : size_(size) {
  if (size > 0) {
    cudaMalloc(&data_, size * sizeof(T));
  }
}

template <DType T>
TensorStorage<T, CUDA>::~TensorStorage() {
  if (data_) {
    cudaFree(data_);
  }
}

template <DType T>
TensorStorage<T, CUDA>::TensorStorage(TensorStorage&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

template <DType T>
void TensorStorage<T, CUDA>::resize(int size) {
  if (data_) { cudaFree(data_); }
  size_ = size;
  if (size > 0) {
    cudaMalloc(&data_, size * sizeof(T));
  }
}

template <DType T>
void TensorStorage<T, CUDA>::fill(T value) {
  if (size_ == 0) { return; }

  int block_size = cuda::get_block_size();
  int grid_size = cuda::get_grid_size(size_, block_size);

  fill_kernel<<<grid_size, block_size>>>(data_, value, size_);
  CUDA_CHECK(cudaGetLastError()); // NOLINT
}

// Explicit instantiations
template class TensorStorage<float, CUDA>;
template class TensorStorage<bfloat16, CUDA>;
template class TensorStorage<int, CUDA>;
};
