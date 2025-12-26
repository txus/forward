#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

template<typename DeviceT>
__global__ void add_kernel(DeviceT* out, DeviceT* tensor_a, DeviceT* tensor_b, size_t n);


extern template __global__ void add_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>*, Cuda<float>*, size_t);
extern template __global__ void add_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>*, Cuda<int>*, size_t);
__global__ void add_kernel_bf16(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor_a, Cuda<bfloat16>* tensor_b, size_t n);

} // namespace tensor::kernels
