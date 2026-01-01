
#include "zip.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

template <typename TIn1, typename TIn2, typename TOut, typename Func>
__global__ void zip_kernel(TOut* out, const TIn1* tensor_a, const TIn2* tensor_b, Func func, size_t last_dim_a_stride, size_t last_dim_b_stride, size_t out_last_dim, size_t n) {
  auto out_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (out_idx < n) {
    size_t a_idx = (last_dim_a_stride == 0) ? (out_idx / out_last_dim) : out_idx;
    size_t b_idx = (last_dim_b_stride == 0) ? (out_idx / out_last_dim) : out_idx;
    out[out_idx] = func(tensor_a[a_idx], tensor_b[b_idx]);
  }
}

template <typename TIn1, typename TIn2, typename TOut, typename Func>
Tensor<TOut, CUDA> zip(const TensorView<TIn1, CUDA>& tensor_a, const TensorView<TIn1, CUDA>& tensor_b, Func func) {
  auto shp_a = tensor_a.shape;
  auto rank_a = shp_a.size();
  auto shp_b = tensor_b.shape;
  auto rank_b = shp_b.size();

  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");
  assert(rank_a == rank_b && "the two tensors should be the same rank");

  auto last_dim_a = shp_a[rank_a-1];
  auto last_dim_b = shp_b[rank_b-1];

  auto last_dim_a_stride = 1;
  auto last_dim_b_stride = 1;

  auto out_shape = shp_a;

  for (size_t idx = 0; idx < rank_a; ++idx) {
    if (idx == rank_a-1) {
      if (shp_a[idx] == 1 && shp_b[idx] != 1) {
        last_dim_a_stride = 0;
        out_shape = shp_b;
      }
      if (shp_b[idx] == 1 && shp_a[idx] != 1) {
        last_dim_b_stride = 0;
        out_shape = shp_a;
      }
    } else {
      assert(shp_a[idx] == shp_b[idx] && "both tensors should have the same dims except the last one");
    }
  }

  size_t n_elements = std::max(tensor_a.data_size, tensor_b.data_size);
  TensorStorage<TOut, CUDA> storage(n_elements);

  Tensor<TOut, CUDA> out{out_shape, std::move(storage)};

  size_t out_last_dim = out_shape[out_shape.size() - 1];

  size_t block_size = cuda::get_block_size(n_elements);
  size_t grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<TOut>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<TIn1>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<TIn2>*>(tensor_b.data); // NOLINT

  zip_kernel<Cuda<TIn1>, Cuda<TIn2>, Cuda<TOut>><<<grid_size, block_size>>>(out_d, a_d, b_d, func, last_dim_a_stride, last_dim_b_stride, out_last_dim, n_elements);

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
