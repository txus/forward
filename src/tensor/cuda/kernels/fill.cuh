#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

// Kernels operate on device-native types via Cuda<T>
template<typename DeviceT>
__global__ void fill_kernel(DeviceT* out, DeviceT value, size_t n);

// Explicit declarations
extern template __global__ void fill_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>, size_t);
extern template __global__ void fill_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>, size_t);
extern template __global__ void fill_kernel<Cuda<bfloat16>>(Cuda<bfloat16>*, Cuda<bfloat16>, size_t);

} // namespace tensor::kernels
