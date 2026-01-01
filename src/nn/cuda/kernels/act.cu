#include "act.cuh"
#include "utils.cuh"
#include <tensor/device_type.hpp>

namespace nn::kernels {

using namespace tensor;
using namespace tensor::dtype;

template <typename TIn, typename TOut, typename Func>
__global__ void map_kernel(TOut* out, const TIn* tensor, Func func, size_t n) {
  auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < n) {
    out[idx] = func(tensor[idx]);
  }
}

template <typename TIn, typename TOut, typename Func>
Tensor<TOut, CUDA> map(const TensorView<TIn, CUDA>& tensor, Func func) {
  assert(tensor.is_contiguous() && "the tensor should be contiguous");

  size_t n_elements = tensor.data_size;
  TensorStorage<TOut, CUDA> storage(n_elements);

  Tensor<TOut, CUDA> out{tensor.shape, std::move(storage)};

  size_t block_size = cuda::get_block_size(n_elements);
  size_t grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<TOut>*>(out.data()); // NOLINT
  auto* input_d = reinterpret_cast<Cuda<TIn>*>(tensor.data); // NOLINT

  map_kernel<Cuda<TIn>, Cuda<TOut>><<<grid_size, block_size>>>(out_d, input_d, func, n_elements);

  return out;
}

// silu

template <typename T>
struct SiLU {
    __device__ T operator()(T value) const { return value / T(expf(-value) + 1.0); } // x * sigmoid(x)
};

template <typename T>
Tensor<T, CUDA> silu(const TensorView<T, CUDA>& tensor) {
  return map<T, T>(tensor, SiLU<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> silu(const TensorView<bfloat16, CUDA>& tensor);

}
