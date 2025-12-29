#pragma once

#include <tensor/tensor.hpp>

namespace tensor::kernels {

template<typename T>
Tensor<T, CUDA> cat(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b, int dim);

} // namespace tensor::kernels
