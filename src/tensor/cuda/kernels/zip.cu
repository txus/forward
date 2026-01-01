
#include "zip.cuh"
#include "broadcast.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

// Kernel with full broadcasting support
template <typename TIn1, typename TIn2, typename TOut, typename Func>
__global__ void zip_broadcast_kernel(TOut* out, const TIn1* tensor_a, const TIn2* tensor_b,
                                      Func func,
                                      const size_t* out_shape, const size_t* a_strides,
                                      const size_t* b_strides, size_t ndim, size_t n) {
  auto out_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (out_idx < n) {
    size_t a_idx = broadcast_index(out_idx, out_shape, a_strides, ndim);
    size_t b_idx = broadcast_index(out_idx, out_shape, b_strides, ndim);
    out[out_idx] = func(tensor_a[a_idx], tensor_b[b_idx]);
  }
}

template <typename TIn1, typename TIn2, typename TOut, typename Func>
Tensor<TOut, CUDA> zip(const TensorView<TIn1, CUDA>& tensor_a, const TensorView<TIn1, CUDA>& tensor_b, Func func) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");

  // Compute broadcast configuration
  auto config = compute_broadcast_config(tensor_a.shape, tensor_b.shape);

  TensorStorage<TOut, CUDA> storage(config.total_elements);
  Tensor<TOut, CUDA> out{config.out_shape, std::move(storage)};

  // Allocate and copy broadcast params to device
  DeviceBroadcastParams params(config);

  size_t block_size = cuda::get_block_size(config.total_elements);
  size_t grid_size = cuda::get_grid_size(config.total_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<TOut>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<const Cuda<TIn1>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<const Cuda<TIn2>*>(tensor_b.data); // NOLINT

  zip_broadcast_kernel<Cuda<TIn1>, Cuda<TIn2>, Cuda<TOut>><<<grid_size, block_size>>>(
      out_d, a_d, b_d, func, params.d_out_shape, params.d_a_strides, params.d_b_strides,
      config.ndim, config.total_elements);

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
