#include "slice.cuh"
#include "utils.cuh"
#include <cstddef>
#include <tensor/device_type.hpp>
#include <cuda_runtime.h>

namespace tensor::kernels {

using namespace dtype;

template <typename T>
__global__ void slice_kernel(Cuda<T>* out, const Cuda<T>* input, size_t start_offset, size_t chunk_size, size_t source_stride) {
  size_t operation_idx = blockIdx.x;

  auto in_base = (operation_idx * source_stride) + start_offset;
  auto out_base = operation_idx * chunk_size;

  for (size_t element = threadIdx.x; element < chunk_size; element += blockDim.x) {
    out[out_base + element] = input[in_base + element];
  }
}

template <typename T>
Tensor<std::remove_const_t<T>, CUDA> slice(const TensorView<T, CUDA>& view, int dim, size_t start, size_t end) {
  using OutT = std::remove_const_t<T>;
  assert(view.is_contiguous() && "tensor should be contiguous");

  auto shape = view.shape;

  if (dim < 0) {
    dim = static_cast<int>(shape.size()) + dim;
  }

  Shape new_shape{shape};
  new_shape[dim] = end - start;

  // product of all dimensions after dim
  size_t inner_stride = 1;
  for (size_t idx = dim + 1; idx < shape.size(); ++idx) {
    inner_stride *= shape[idx];
  }

  // product of all dimensions before dim
  size_t outer_iterations = 1;
  for (size_t idx = 0; idx < static_cast<size_t>(dim); ++idx) {
    outer_iterations *= shape[idx];
  }

  size_t source_stride = shape[dim] * inner_stride;
  size_t chunk_size = (end - start) * inner_stride;
  size_t start_offset = start * inner_stride;

  size_t n_elements = outer_iterations * chunk_size;
  TensorStorage<OutT, CUDA> storage(n_elements);
  Tensor<OutT, CUDA> out{new_shape, std::move(storage)};

  // fast path: if slicing on first dimension, just use cudaMemcpy
  if (dim == 0) {
    size_t bytes = n_elements * sizeof(T);
    CUDA_CHECK(cudaMemcpy(out.data(), view.data + start_offset, bytes, cudaMemcpyDeviceToDevice)); // NOLINT
    return out;
  }

  size_t block_size = cuda::get_block_size(chunk_size);

  auto* out_d = reinterpret_cast<Cuda<OutT>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<const Cuda<OutT>*>(view.data); // NOLINT

  slice_kernel<OutT><<<outer_iterations, block_size>>>(out_d, in_d, start_offset, chunk_size, source_stride);

  return out;
}

template Tensor<bfloat16, CUDA> slice(const TensorView<bfloat16, CUDA>& view, int dim, size_t start, size_t end);
template Tensor<bfloat16, CUDA> slice(const TensorView<const bfloat16, CUDA>& view, int dim, size_t start, size_t end);
template Tensor<float, CUDA> slice(const TensorView<float, CUDA>& view, int dim, size_t start, size_t end);
template Tensor<float, CUDA> slice(const TensorView<const float, CUDA>& view, int dim, size_t start, size_t end);
template Tensor<int, CUDA> slice(const TensorView<int, CUDA>& view, int dim, size_t start, size_t end);

} // namespace tensor::kernels
