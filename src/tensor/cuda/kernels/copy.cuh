#pragma once

#include <tensor/tensor.hpp>
#include <tensor/device_type.hpp>

namespace tensor::kernels {

template<typename T>
Tensor<T, CUDA> copy(const TensorView<T, CUDA>& view);

template<typename TIn, typename TOut>
Tensor<TOut, CUDA> to(const TensorView<TIn, CUDA>& view);

} // namespace tensor::kernels
