#include "map.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

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

// pow

template <typename T>
struct PowValueScalar {
    T scalar;
    __device__ T operator()(T value) const { return powf(value, scalar); }
};

template <typename T>
struct PowScalarValue {
    T scalar;
    __device__ T operator()(T value) const { return powf(scalar, value); }
};

template <typename T>
Tensor<T, CUDA> pow_tensor_scalar(const TensorView<T, CUDA>& tensor, T scalar) {
  return map<T, T>(tensor, PowValueScalar{to_device_type(scalar, CUDA{})});
};

template <typename T>
Tensor<T, CUDA> pow_scalar_tensor(T scalar, const TensorView<T, CUDA>& tensor) {
  return map<T, T>(tensor, PowScalarValue{to_device_type(scalar, CUDA{})});
};

template Tensor<bfloat16, CUDA> pow_tensor_scalar(const TensorView<bfloat16, CUDA>& tensor, bfloat16 scalar);
template Tensor<bfloat16, CUDA> pow_scalar_tensor(bfloat16 scalar, const TensorView<bfloat16, CUDA>& tensor);

template Tensor<float, CUDA> pow_tensor_scalar(const TensorView<float, CUDA>& tensor, float scalar);
template Tensor<float, CUDA> pow_scalar_tensor(float scalar, const TensorView<float, CUDA>& tensor);

// cos

template <typename T>
struct Cos {
    __device__ T operator()(T value) const { return cosf(value); }
};

template <typename T>
Tensor<T, CUDA> cos(const TensorView<T, CUDA>& tensor) {
  return map<T, T>(tensor, Cos<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> cos(const TensorView<bfloat16, CUDA>& tensor);
template Tensor<float, CUDA> cos(const TensorView<float, CUDA>& tensor);

// sin

template <typename T>
struct Sin {
    __device__ T operator()(T value) const { return sinf(value); }
};

template <typename T>
Tensor<T, CUDA> sin(const TensorView<T, CUDA>& tensor) {
  return map<T, T>(tensor, Sin<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> sin(const TensorView<bfloat16, CUDA>& tensor);
template Tensor<float, CUDA> sin(const TensorView<float, CUDA>& tensor);

// exp

template <typename T>
struct Exp {
    __device__ T operator()(T value) const { return expf(value); }
};

template <typename T>
Tensor<T, CUDA> exp(const TensorView<T, CUDA>& tensor) {
  return map<T, T>(tensor, Exp<Cuda<T>>{});
};

template Tensor<bfloat16, CUDA> exp(const TensorView<bfloat16, CUDA>& tensor);
template Tensor<float, CUDA> exp(const TensorView<float, CUDA>& tensor);
template Tensor<int, CUDA> exp(const TensorView<int, CUDA>& tensor);

// div

template <typename T>
struct Div {
    T scalar;
    __device__ T operator()(T value) const { return value / scalar; }
};

template <typename T>
Tensor<T, CUDA> div(const TensorView<T, CUDA>& tensor, T scalar) {
  return map<T, T>(tensor, Div<Cuda<T>>{to_device_type(scalar, CUDA{})});
};

template Tensor<bfloat16, CUDA> div(const TensorView<bfloat16, CUDA>& tensor, bfloat16 scalar);
template Tensor<float, CUDA> div(const TensorView<float, CUDA>& tensor, float scalar);
template Tensor<int, CUDA> div(const TensorView<int, CUDA>& tensor, int scalar);

// mul

template <typename T>
struct Mul {
    T scalar;
    __device__ T operator()(T value) const { return value * scalar; }
};

template <typename T>
Tensor<T, CUDA> mul(const TensorView<T, CUDA>& tensor, T scalar) {
  return map<T, T>(tensor, Mul<Cuda<T>>{to_device_type(scalar, CUDA{})});
};

template Tensor<bfloat16, CUDA> mul(const TensorView<bfloat16, CUDA>& tensor, bfloat16 scalar);
template Tensor<float, CUDA> mul(const TensorView<float, CUDA>& tensor, float scalar);
template Tensor<int, CUDA> mul(const TensorView<int, CUDA>& tensor, int scalar);
}
