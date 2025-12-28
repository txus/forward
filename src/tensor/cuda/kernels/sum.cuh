#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void sum_float_kernel(Cuda<float>* out, Cuda<float>* input, size_t num_reductions, size_t reduce_size, size_t reduce_stride);

Tensor<float, CUDA> sum_float(const TensorView<float, CUDA>& input, int dim, bool keepdim);

} // namespace tensor::kernels
