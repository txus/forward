#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>

namespace nn::kernels {

using namespace tensor;

template<typename T>
Tensor<T, CUDA> silu(const TensorView<T, CUDA>& tensor);

} // namespace nn::kernels
