#pragma once

#include <tensor/tensor.hpp>
#include <tensor/device_type.hpp>

namespace tensor::kernels {

using namespace dtype;

template <typename T> Tensor<T, CUDA> tril(const TensorView<T, CUDA>& tensor, bool diagonal);

} // namespace tensor::kernels
