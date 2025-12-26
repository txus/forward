#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void div_float_kernel(Cuda<float>* out, const Cuda<float>* tensor_a, const Cuda<float>* tensor_b, size_t n);
__global__ void div_scalar_float_kernel(Cuda<float>* out, const Cuda<float>* tensor_a, float tensor_b, size_t n);

Tensor<float, CUDA> div_float(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b);
Tensor<float, CUDA> div_float(const TensorView<float, CUDA>& tensor_a, float scalar);

} // namespace tensor::kernels
