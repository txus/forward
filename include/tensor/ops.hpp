#pragma once

#include <tensor/tensor.hpp>

namespace tensor {

template <DType T, Device D> Tensor<T, D> arange(T start, T end, T step);
template <DType T, Device D>
Tensor<T, D> add(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b);
template <DType T, Device D>
Tensor<T, D> mul(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b);
template <DType T, Device D> Tensor<T, D> mul(const TensorView<T, D>& tensor_a, T scalar);

template <DType T, Device D> Tensor<T, D> pow(T scalar, const TensorView<T, D>& tensor);
template <DType T, Device D> Tensor<T, D> pow(const TensorView<T, D>& tensor, T scalar);

template <DType T, Device D>
Tensor<T, D> matmul(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b);

template <DType T, Device D>
Tensor<T, D> cat(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b, int dim);

template <DType T, Device D>
Tensor<T, D> slice(const TensorView<T, D>& view, int dim, size_t start, size_t end);
template <DType T, Device D>
Tensor<const T, D> slice(const TensorView<const T, D>& view, int dim, size_t start, size_t end);

} // namespace tensor
