#pragma once

#include <tensor/tensor.hpp>

namespace tensor::kernels {

template<typename T>
Tensor<T, CUDA> matmul(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b);

} // namespace tensor::kernels
