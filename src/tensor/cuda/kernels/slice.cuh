#pragma once

#include <tensor/tensor.hpp>

namespace tensor::kernels {

template<typename T>
Tensor<std::remove_const_t<T>, CUDA> slice(const TensorView<T, CUDA>& view, int dim, size_t start, size_t end);

} // namespace tensor::kernels
