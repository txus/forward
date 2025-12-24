#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

// Kernels operate on device-native types via Cuda<T>
template<typename DeviceT>
__global__ void arange_kernel(DeviceT* out, DeviceT start, DeviceT end, DeviceT step, size_t n);

// Explicit declarations
extern template __global__ void arange_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>, Cuda<float>, Cuda<float>, size_t);
extern template __global__ void arange_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>, Cuda<int>, Cuda<int>, size_t);
extern template __global__ void arange_kernel<Cuda<bfloat16>>(Cuda<bfloat16>*, Cuda<bfloat16>, Cuda<bfloat16>, Cuda<bfloat16>, size_t);

} // namespace tensor::kernels
