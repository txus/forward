#pragma once

#include <tensor/tensor.hpp>

namespace tensor {

template <DType T, Device D>
Tensor<T, D> add(TensorView<T, D> &a, TensorView<T, D> &b);

} // namespace tensor
