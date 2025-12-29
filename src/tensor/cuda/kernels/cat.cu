#include "cat.cuh"
#include "utils.cuh"
#include <cstddef>
#include <tensor/device_type.hpp>
#include <cuda_runtime.h>

namespace tensor::kernels {

using namespace dtype;

template <typename T>
__global__ void cat_kernel(Cuda<T>* out, Cuda<T>* tensor_a, const Cuda<T>* tensor_b, size_t inner_size_a, size_t inner_size_b) {
  // there are as many blocks as operations to do, which one are we doing?
  size_t operation_idx = blockIdx.x;
  size_t out_inner_size = inner_size_a + inner_size_b;

  auto a_base = operation_idx * inner_size_a;
  auto b_base = operation_idx * inner_size_b;
  auto out_base = operation_idx * out_inner_size;

  for (size_t element = threadIdx.x; element < out_inner_size; element += blockDim.x) {
    if (element < inner_size_a) {
      out[out_base + element] = tensor_a[a_base + element];
    } else {
      out[out_base + element] = tensor_b[b_base + (element - inner_size_a)];
    }
  }
}

template<typename T>
inline void fast_cat(T* out, T* in1, T* in2, size_t n_elements_a, size_t n_elements_b) {
    size_t a_bytes = n_elements_a * sizeof(T);
    size_t b_bytes = n_elements_b * sizeof(T);
    CUDA_CHECK(cudaMemcpy(out, in1, a_bytes, cudaMemcpyDeviceToDevice)); // NOLINT
    CUDA_CHECK(cudaMemcpy(out + n_elements_a, in2, b_bytes, cudaMemcpyDeviceToDevice)); // NOLINT
}

template <typename T>
Tensor<T, CUDA> cat(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b, int dim) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");

  bool is_first_dim = dim == 0;

  if (dim == -1) {
    dim = static_cast<int>(tensor_a.shape[tensor_a.shape.size()-1]);
  }

  Shape new_shape{};
  for (size_t idx = 0; idx < tensor_a.shape.size(); ++idx) {
    if (dim == idx) {
      new_shape.push_back(tensor_a.shape[idx] + tensor_b.shape[idx]);
    } else {
      assert(tensor_a.shape[idx] == tensor_b.shape[idx] && "tensors should have the same dimensions except for the catting dimension");
      new_shape.push_back(tensor_a.shape[idx]);
    }
  }

  size_t n_elements = tensor_a.data_size + tensor_b.data_size;
  TensorStorage<T, CUDA> storage(n_elements);
  Tensor<T, CUDA> out{new_shape, std::move(storage)};

  if (is_first_dim) { // we're in luck, just copy tensor b right after tensor a
    fast_cat<T>(out.data(), tensor_a.data, tensor_b.data, tensor_a.data_size, tensor_b.data_size);
    return out;
  }

  size_t outer_size = 1;
  for (size_t idx = 0; idx < dim; ++idx) {
    outer_size *= tensor_a.shape[idx];
  }

  size_t inner_size_a = 1;
  size_t inner_size_b = 1;
  for (size_t idx = dim; idx < tensor_a.shape.size(); ++idx) {
    inner_size_a *= tensor_a.shape[idx];
    inner_size_b *= tensor_b.shape[idx];
  }

  size_t block_size = cuda::get_block_size(inner_size_a + inner_size_b);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<T>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<T>*>(tensor_b.data); // NOLINT

  cat_kernel<T><<<outer_size, block_size>>>(out_d, a_d, b_d, inner_size_a, inner_size_b);

  return out;
}


template Tensor<bfloat16, CUDA> cat(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b, int dim);
template Tensor<float, CUDA> cat(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b, int dim);

} // namespace tensor::kernels
