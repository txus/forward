#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fmt/core.h>

#include <tensor/ops.hpp>
#include <tensor/device_type.hpp>

#include "kernels/arange.cuh"
#include "kernels/add.cuh"
#include "kernels/sub.cuh"
#include "kernels/div.cuh"
#include "kernels/mul.cuh"
#include "kernels/utils.cuh"

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

template <>
void replace_from_(Tensor<bfloat16, CUDA>& out, const TensorView<bfloat16, CUDA>& in) {
  CUDA_CHECK(cudaMemcpy(out.data(), in.data, in.data_size * sizeof(bfloat16), cudaMemcpyDeviceToDevice)); // NOLINT
}

template <>
Tensor<bfloat16, CUDA> add(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b) {
  return add_bfloat16(tensor_a, tensor_b);
}

template <>
Tensor<float, CUDA> sub(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b) {
  return sub_float(tensor_a, tensor_b);
}

template <>
Tensor<float, CUDA> div(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b) {
  return div_float(tensor_a, tensor_b);
}

template <>
Tensor<float, CUDA> div(const TensorView<float, CUDA>& tensor_a, float scalar) {
  return div_float(tensor_a, scalar);
}

template <>
Tensor<bfloat16, CUDA> mul(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b) {
  return mul_bfloat16(tensor_a, tensor_b);
}

template <>
Tensor<bfloat16, CUDA> mul(const TensorView<bfloat16, CUDA>& tensor_a, bfloat16 scalar) {
  return mul_bfloat16(tensor_a, scalar);
}

} // namespace tensor
