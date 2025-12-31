#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>

namespace tensor::kernels {

using namespace dtype;

template<typename T>
Tensor<T, CUDA> pow_tensor_scalar(const TensorView<T, CUDA>& tensor, T scalar);
template<typename T>
Tensor<T, CUDA> pow_scalar_tensor(T scalar, const TensorView<T, CUDA>& tensor);

template<typename T>
Tensor<T, CUDA> sin(const TensorView<T, CUDA>& tensor);
template<typename T>
Tensor<T, CUDA> cos(const TensorView<T, CUDA>& tensor);
template<typename T>
Tensor<T, CUDA> exp(const TensorView<T, CUDA>& tensor);

template<typename T>
Tensor<T, CUDA> div(const TensorView<T, CUDA>& tensor, T scalar);
template<typename T>
Tensor<T, CUDA> mul(const TensorView<T, CUDA>& tensor, T scalar);

} // namespace tensor::kernels
