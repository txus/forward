#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>

namespace nn::kernels {

using namespace tensor;

template<typename T>
Tensor<T, CUDA> rms_norm(const TensorView<T, CUDA>& input, const TensorView<const T, CUDA>& weights, T eps);

} // namespace nn::kernels
