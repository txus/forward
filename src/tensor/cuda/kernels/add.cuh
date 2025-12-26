#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void add_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor_a, Cuda<bfloat16>* tensor_b, size_t n);

Tensor<bfloat16, CUDA> add_bfloat16(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b);

} // namespace tensor::kernels
