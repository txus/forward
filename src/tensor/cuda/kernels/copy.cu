#include "copy.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

template<typename TIn, typename TOut>
__global__ void to_kernel(TOut* out, TIn* input, size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < n) {
    out[idx] = TOut(input[idx]);
  }
}

template <typename TIn, typename TOut> Tensor<TOut, CUDA> to(const TensorView<TIn, CUDA>& view) {
  auto n_elements = view.data_size;

  TensorStorage<TOut, CUDA> storage(n_elements);

  Tensor<TOut, CUDA> out{view.shape, std::move(storage)};

  size_t block_size = cuda::get_block_size(n_elements);
  size_t grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<TOut>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<Cuda<TIn>*>(view.data); // NOLINT

  to_kernel<Cuda<TIn>, Cuda<TOut>><<<grid_size, block_size>>>(out_d, in_d, n_elements);

  return out;
}

template Tensor<bfloat16, CUDA> to(const TensorView<float, CUDA>& view);
template Tensor<float, CUDA> to(const TensorView<bfloat16, CUDA>& view);

template <typename T>
Tensor<T, CUDA> copy(const TensorView<T, CUDA>& view) {
  auto n_elements = view.data_size;
  TensorStorage<T, CUDA> storage(n_elements);
  Tensor<T, CUDA> out{view.shape, std::move(storage)};

  CUDA_CHECK(cudaMemcpy(out.data(), view.data, n_elements * sizeof(T), cudaMemcpyDeviceToDevice)); // NOLINT
  return out;
}

template Tensor<bfloat16, CUDA> copy(const TensorView<bfloat16, CUDA>& view);
template Tensor<float, CUDA> copy(const TensorView<float, CUDA>& view);
template Tensor<int, CUDA> copy(const TensorView<int, CUDA>& view);

} // namespace tensor::kernels
