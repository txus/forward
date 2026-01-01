#include "copy.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

// Strided copy kernel - converts logical index to strided source index
template<typename TOut, typename TIn>
__global__ void copy_strided_kernel(TOut* out, const TIn* input,
                                     const size_t* strides, const size_t* shape,
                                     size_t ndim, size_t n) {
  size_t out_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (out_idx < n) {
    // Convert flat output index to strided input index
    size_t in_idx = 0;
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; ++d) {
      size_t dim_size = shape[d];
      size_t coord = remaining / (n / dim_size);
      remaining = remaining % (n / dim_size);
      n = n / dim_size;
      in_idx += coord * strides[d];
    }
    out[out_idx] = input[in_idx];
  }
}

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
template Tensor<float, CUDA> to(const TensorView<int, CUDA>& view);

// Check if view is contiguous (strides match what you'd expect for row-major layout)
template <typename T>
bool is_contiguous(const TensorView<T, CUDA>& view) {
  if (view.shape.empty()) return true;

  size_t expected_stride = 1;
  for (int d = static_cast<int>(view.shape.size()) - 1; d >= 0; --d) {
    if (view.stride[d] != expected_stride) return false;
    expected_stride *= view.shape[d];
  }
  return true;
}

template <typename T>
Tensor<std::remove_const_t<T>, CUDA> copy(const TensorView<T, CUDA>& view) {
  using OutT = std::remove_const_t<T>;
  auto n_elements = view.data_size;
  TensorStorage<OutT, CUDA> storage(n_elements);
  Tensor<OutT, CUDA> out{view.shape, std::move(storage)};

  if (is_contiguous(view)) {
    // Fast path: contiguous data, use cudaMemcpy
    CUDA_CHECK(cudaMemcpy(out.data(), view.data, n_elements * sizeof(T), cudaMemcpyDeviceToDevice)); // NOLINT
  } else {
    // Slow path: strided data, use kernel
    size_t ndim = view.shape.size();

    // Copy strides and shape to device
    size_t* d_strides;
    size_t* d_shape;
    CUDA_CHECK(cudaMalloc(&d_strides, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_strides, view.stride.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape, view.shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t block_size = cuda::get_block_size(n_elements);
    size_t grid_size = cuda::get_grid_size(n_elements, block_size);

    auto* out_d = reinterpret_cast<Cuda<OutT>*>(out.data()); // NOLINT
    auto* in_d = reinterpret_cast<const Cuda<OutT>*>(view.data); // NOLINT

    copy_strided_kernel<Cuda<OutT>, Cuda<OutT>><<<grid_size, block_size>>>(out_d, in_d, d_strides, d_shape, ndim, n_elements);

    CUDA_CHECK(cudaFree(d_strides));
    CUDA_CHECK(cudaFree(d_shape));
  }
  return out;
}

template Tensor<bfloat16, CUDA> copy(const TensorView<bfloat16, CUDA>& view);
template Tensor<bfloat16, CUDA> copy(const TensorView<const bfloat16, CUDA>& view);
template Tensor<float, CUDA> copy(const TensorView<float, CUDA>& view);
template Tensor<float, CUDA> copy(const TensorView<const float, CUDA>& view);
template Tensor<int, CUDA> copy(const TensorView<int, CUDA>& view);
template Tensor<int, CUDA> copy(const TensorView<const int, CUDA>& view);

} // namespace tensor::kernels
