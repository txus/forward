#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <tensor/tensor.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

__global__ void sub_float_kernel(Cuda<float>* out, Cuda<float>* tensor_a, Cuda<float>* tensor_b, size_t n);

Tensor<float, CUDA> sub_float(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b);

} // namespace tensor::kernels
