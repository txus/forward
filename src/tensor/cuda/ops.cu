#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fmt/core.h>

#include <tensor/ops.hpp>
#include <tensor/device_type.hpp>

#include "kernels/arange.cuh"
#include "kernels/sum.cuh"
#include "kernels/max.cuh"
#include "kernels/argmax.cuh"
#include "kernels/masked_fill.cuh"
#include "kernels/cat.cuh"
#include "kernels/map.cuh"
#include "kernels/zip.cuh"
#include "kernels/tril.cuh"
#include "kernels/slice.cuh"
#include "kernels/utils.cuh"

namespace tensor {

using namespace dtype;
using namespace device;

template <> Tensor<bfloat16, CUDA> arange(bfloat16 start, bfloat16 end, bfloat16 step) {
  return kernels::arange(start, end, step);
}
template <> Tensor<float, CUDA> arange(float start, float end, float step) {
  return kernels::arange(start, end, step);
}
template <> Tensor<int, CUDA> arange(int start, int end, int step) {
  return kernels::arange(start, end, step);
}

template <typename T, typename D>
void replace_from_(Tensor<T, D>& out, const TensorView<T, D>& input) {
  CUDA_CHECK(cudaMemcpy(out.data(), input.data, input.data_size * sizeof(T), cudaMemcpyDeviceToDevice)); // NOLINT
}

template void replace_from_(Tensor<bfloat16, CUDA>& out, const TensorView<bfloat16, CUDA>& input);
template void replace_from_(Tensor<float, CUDA>& out, const TensorView<float, CUDA>& input);
template void replace_from_(Tensor<int, CUDA>& out, const TensorView<int, CUDA>& input);

template <>
Tensor<bfloat16, CUDA> add(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b) {
  return kernels::add(tensor_a, tensor_b);
}

template <>
Tensor<float, CUDA> sub(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b) {
  return kernels::sub(tensor_a, tensor_b);
}

template <>
Tensor<float, CUDA> div(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b) {
  return kernels::div(tensor_a, tensor_b);
}

template <>
Tensor<float, CUDA> div(const TensorView<float, CUDA>& tensor_a, float scalar) {
  return kernels::div(tensor_a, scalar);
}

template <>
Tensor<bfloat16, CUDA> mul(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b) {
  return kernels::mul(tensor_a, tensor_b);
}

template <>
Tensor<bfloat16, CUDA> mul(const TensorView<bfloat16, CUDA>& tensor_a, bfloat16 scalar) {
  return kernels::mul(tensor_a, scalar);
}

template <>
Tensor<float, CUDA> sum(const TensorView<float, CUDA>& input, int dim, bool keepdim) {
  return kernels::sum_float(input, dim, keepdim);
}

template <>
Tensor<float, CUDA> max(const TensorView<float, CUDA>& input, int dim, bool keepdim) {
  return kernels::max_float(input, dim, keepdim);
}

template <>
Tensor<int, CUDA> argmax(const TensorView<bfloat16, CUDA>& input, int dim, bool keepdim) {
  return kernels::argmax_bfloat16(input, dim, keepdim);
}

template <>
Tensor<bfloat16, CUDA> masked_fill(const TensorView<bfloat16, CUDA>& input, const TensorView<int, CUDA>& mask, bfloat16 masked_value) {
  return kernels::masked_fill_bfloat16(input, mask, masked_value);
}

template <>
Tensor<bfloat16, CUDA> cat(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b, int dim) {
  return kernels::cat(tensor_a, tensor_b, dim);
}

template <>
Tensor<float, CUDA> cat(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b, int dim) {
  return kernels::cat(tensor_a, tensor_b, dim);
}

template <>
Tensor<float, CUDA> pow(const TensorView<float, CUDA>& tensor, float scalar) {
  return kernels::pow_tensor_scalar(tensor, scalar);
}

template <>
Tensor<float, CUDA> pow(float scalar, const TensorView<float, CUDA>& tensor) {
  return kernels::pow_scalar_tensor(scalar, tensor);
}

template <>
Tensor<float, CUDA> cos(const TensorView<float, CUDA>& tensor) {
  return kernels::cos(tensor);
}

template <>
Tensor<float, CUDA> sin(const TensorView<float, CUDA>& tensor) {
  return kernels::sin(tensor);
}

template <>
Tensor<float, CUDA> exp(const TensorView<float, CUDA>& tensor) {
  return kernels::exp(tensor);
}

template <>
Tensor<bfloat16, CUDA> tril(const TensorView<bfloat16, CUDA>& tensor, bool diagonal) {
  return kernels::tril(tensor, diagonal);
}

template <>
Tensor<int, CUDA> tril(const TensorView<int, CUDA>& tensor, bool diagonal) {
  return kernels::tril(tensor, diagonal);
}

template <>
Tensor<bfloat16, CUDA> slice(const TensorView<bfloat16, CUDA>& view, int dim, size_t start, size_t end) {
  return kernels::slice(view, dim, start, end);
}

template <>
Tensor<float, CUDA> slice(const TensorView<float, CUDA>& view, int dim, size_t start, size_t end) {
  return kernels::slice(view, dim, start, end);
}

} // namespace tensor
