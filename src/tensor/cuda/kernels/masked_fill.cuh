#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void masked_fill_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* input, Cuda<bfloat16>* mask, bfloat16 masked_value);

Tensor<bfloat16, CUDA> masked_fill_bfloat16(const TensorView<bfloat16, CUDA>& input, const TensorView<int, CUDA>& mask, bfloat16 masked_value);

} // namespace tensor::kernels
