#pragma once

#include <tensor/tensor.hpp>
#include <tensor/device_type.hpp>

namespace tensor::kernels {

using namespace dtype;

template <typename T> Tensor<T, CUDA> arange(T start, T end, T step);

} // namespace tensor::kernels
