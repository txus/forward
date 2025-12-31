
#include "zip.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

template <typename TIn1, typename TIn2, typename TOut, typename Func>
__global__ void zip_kernel(TOut* out, const TIn2* tensor_a, const TIn2* tensor_b, Func func, size_t n) {
  auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < n) {
    out[idx] = func(tensor_a[idx], tensor_b[idx]);
  }
}

template <typename TIn1, typename TIn2, typename TOut, typename Func>
Tensor<TOut, CUDA> zip(const TensorView<TIn1, CUDA>& tensor_a, const TensorView<TIn1, CUDA>& tensor_b, Func func) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");
  assert(tensor_a.shape == tensor_b.shape && "the two tensors should be the same shape");

  size_t n_elements = tensor_a.data_size;
  TensorStorage<TOut, CUDA> storage(n_elements);

  Tensor<TOut, CUDA> out{tensor_a.shape, std::move(storage)};

  size_t block_size = cuda::get_block_size(n_elements);
  size_t grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<TOut>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<TIn1>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<TIn2>*>(tensor_b.data); // NOLINT

  zip_kernel<Cuda<TIn1>, Cuda<TIn2>, Cuda<TOut>><<<grid_size, block_size>>>(out_d, a_d, b_d, func, n_elements);

  return out;
}

template <typename T>
struct Add {
    __device__ T operator()(T value1, T value2) const { return value1 + value2; }
};

template <typename T>
Tensor<T, CUDA> add(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b) {
  return zip<T, T, T>(tensor_a, tensor_b, Add<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> add(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b);
template Tensor<float, CUDA> add(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b);
template Tensor<int, CUDA> add(const TensorView<int, CUDA>& tensor_a, const TensorView<int, CUDA>& tensor_b);

template <typename T>
struct Sub {
    __device__ T operator()(T value1, T value2) const { return value1 - value2; }
};

template <typename T>
Tensor<T, CUDA> sub(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b) {
  return zip<T, T, T>(tensor_a, tensor_b, Sub<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> sub(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b);
template Tensor<float, CUDA> sub(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b);
template Tensor<int, CUDA> sub(const TensorView<int, CUDA>& tensor_a, const TensorView<int, CUDA>& tensor_b);

template <typename T>
struct Mul {
    __device__ T operator()(T value1, T value2) const { return value1 * value2; }
};

template <typename T>
Tensor<T, CUDA> mul(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b) {
  return zip<T, T, T>(tensor_a, tensor_b, Mul<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> mul(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b);
template Tensor<float, CUDA> mul(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b);
template Tensor<int, CUDA> mul(const TensorView<int, CUDA>& tensor_a, const TensorView<int, CUDA>& tensor_b);

template <typename T>
struct Div {
    __device__ T operator()(T value1, T value2) const { return value1 / value2; }
};

template <typename T>
Tensor<T, CUDA> div(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b) {
  return zip<T, T, T>(tensor_a, tensor_b, Div<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> div(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b);
template Tensor<float, CUDA> div(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b);
template Tensor<int, CUDA> div(const TensorView<int, CUDA>& tensor_a, const TensorView<int, CUDA>& tensor_b);

}
