#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fmt/core.h>

#include <tensor/ops.hpp>
#include <tensor/device_type.hpp>

#include "kernels/arange.cuh"
#include "kernels/add.cuh"
#include "kernels/sub.cuh"
#include "utils.cuh"

namespace tensor {

using namespace dtype;
using namespace device;
using namespace kernels;

template <typename T, typename D> Tensor<T, D> arange(T start, T end, T step) {
  auto n_elements = static_cast<size_t>((end - start) / step);

  TensorStorage<T, D> storage(n_elements);

  Shape shape{n_elements};

  Tensor<T, D> out{shape, std::move(storage)};

  int block_size = cuda::get_block_size();
  int grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* device_data = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  Cuda<T> device_start = to_device_type(start, D{});
  Cuda<T> device_end = to_device_type(end, D{});
  Cuda<T> device_step = to_device_type(step, D{});

  arange_kernel<<<grid_size, block_size>>>(device_data, device_start, device_end, device_step, n_elements);

  return out;
}

template Tensor<int, CUDA> arange(int start, int end, int step);
template Tensor<float, CUDA> arange(float start, float end, float step);
template Tensor<bfloat16, CUDA> arange(bfloat16 start, bfloat16 end, bfloat16 step);

template <typename T, typename D, typename Func>
Tensor<std::remove_const_t<T>, D> element_wise(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b, Func func) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");
  assert(tensor_a.shape == tensor_b.shape && "the two tensors should be the same shape");

  auto n_elements = tensor_a.data_size;
  TensorStorage<std::remove_const_t<T>, D> storage(n_elements);

  Tensor<std::remove_const_t<T>, D> out{tensor_a.shape, std::move(storage)};

  //int block_size = cuda::get_block_size();
  int block_size = 512;
  int grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<T>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<T>*>(tensor_b.data); // NOLINT

  func<<<grid_size, block_size>>>(out_d, a_d, b_d, n_elements);

  return out;
}

template <typename T, typename D, typename Func>
Tensor<std::remove_const_t<T>, D> element_wise_scalar(const TensorView<T, D>& tensor, std::remove_const_t<T> scalar, Func func) {
  assert(tensor.is_contiguous() && "the two tensors should be contiguous");

  auto n_elements = tensor.data_size;
  TensorStorage<std::remove_const_t<T>, D> storage(n_elements);

  Tensor<std::remove_const_t<T>, D> out{tensor.shape, std::move(storage)};

  int block_size = cuda::get_block_size();
  int grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<T>*>(tensor.data); // NOLINT
  Cuda<T> device_scalar = to_device_type(scalar, D{});

  func<<<grid_size, block_size>>>(out_d, a_d, device_scalar, n_elements);

  return out;
}

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> add(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise<T, D>(tensor_a, tensor_b, add_kernel<Cuda<T>>);
}

template <>
Tensor<bfloat16, CUDA> add(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");
  assert(tensor_a.shape == tensor_b.shape && "the two tensors should be the same shape");

  unsigned int n_elements = tensor_a.data_size;
  TensorStorage<std::remove_const_t<bfloat16>, CUDA> storage(n_elements);

  Tensor<std::remove_const_t<bfloat16>, CUDA> out{tensor_a.shape, std::move(storage)};

  int block_size = 512;
  // each thread handles 8 elements
  int grid_size = cuda::get_grid_size(n_elements / 8, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<bfloat16>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<bfloat16>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<bfloat16>*>(tensor_b.data); // NOLINT

  add_kernel_bf16<<<grid_size, block_size>>>(out_d, a_d, b_d, n_elements);

  return out;
}

template Tensor<float, CUDA> add(const TensorView<float, CUDA>&,
                                 const TensorView<float, CUDA>&);
template Tensor<int, CUDA> add(const TensorView<int, CUDA>&, const TensorView<int, CUDA>&);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> sub(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise<T, D>(tensor_a, tensor_b, sub_kernel<Cuda<T>>);
}

template Tensor<bfloat16, CUDA> sub(const TensorView<bfloat16, CUDA>&,
                                    const TensorView<bfloat16, CUDA>&);
template Tensor<float, CUDA> sub(const TensorView<float, CUDA>&,
                                 const TensorView<float, CUDA>&);
template Tensor<int, CUDA> sub(const TensorView<int, CUDA>&, const TensorView<int, CUDA>&);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> sub(const TensorView<T, D>& tensor, std::remove_const_t<T> scalar) {
  return element_wise_scalar<T, D>(tensor, scalar, sub_scalar_kernel<Cuda<T>>);
}

template Tensor<bfloat16, CUDA> sub(const TensorView<bfloat16, CUDA>&, bfloat16);
template Tensor<float, CUDA> sub(const TensorView<float, CUDA>&, float);
template Tensor<int, CUDA> sub(const TensorView<int, CUDA>&, int);

} // namespace tensor
