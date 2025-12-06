#pragma once

#include <tensor/tensor.hpp>

namespace tensor {

template <Device D> Tensor<int, D> arange(int start, int end, int step = 1);
template <Device D> Tensor<float, D> arange(float start, float end, float step = 1.0);
template <DType T, Device D> Tensor<T, D> add(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b);
template <DType T, Device D> Tensor<T, D> mul(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b);

template <DType T, Device D> Tensor<T, D> pow(T scalar, const TensorView<T, D>& tensor);
template <DType T, Device D> Tensor<T, D> pow(const TensorView<T, D>& tensor, T scalar);

template <DType T, Device D>
Tensor<T, D> matmul(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b);

template <DType T, Device D>
Tensor<T, D> cat(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b, size_t dim);

} // namespace tensor
