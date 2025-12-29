#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void argmax_bfloat16_kernel(int* out, Cuda<bfloat16>* input, size_t num_reductions, size_t reduce_size, size_t reduce_stride);

Tensor<int, CUDA> argmax_bfloat16(const TensorView<bfloat16, CUDA>& input, int dim, bool keepdim);

} // namespace tensor::kernels
