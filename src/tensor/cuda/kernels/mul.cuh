#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void mul_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor_a, Cuda<bfloat16>* tensor_b, size_t n);
__global__ void mul_scalar_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor_a, bfloat16 tensor_b, size_t n);

Tensor<bfloat16, CUDA> mul_bfloat16(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b);
Tensor<bfloat16, CUDA> mul_bfloat16(const TensorView<bfloat16, CUDA>& tensor_a, bfloat16 scalar);

} // namespace tensor::kernels
