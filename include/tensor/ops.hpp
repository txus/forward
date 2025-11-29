#pragma once

#include <tensor/tensor.hpp>

namespace tensor {

template <DType T, Device D> Tensor<T, D> add(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b);

template <DType T, Device D>
Tensor<T, D> matmul(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b);

} // namespace tensor
