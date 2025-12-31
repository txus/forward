#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>

namespace tensor::kernels {

using namespace dtype;

template<typename T>
Tensor<T, CUDA> add(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b);
template<typename T>
Tensor<T, CUDA> sub(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b);
template<typename T>
Tensor<T, CUDA> mul(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b);
template<typename T>
Tensor<T, CUDA> div(const TensorView<T, CUDA>& tensor_a, const TensorView<T, CUDA>& tensor_b);

} // namespace tensor::kernels
