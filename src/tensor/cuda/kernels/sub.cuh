#pragma once

#include <cuda_runtime.h>
#include <tensor/device_type.hpp>
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

template<typename DeviceT>
__global__ void sub_kernel(DeviceT* out, DeviceT* tensor_a, DeviceT* tensor_b, size_t n);

extern template __global__ void sub_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>*, Cuda<float>*, size_t);
extern template __global__ void sub_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>*, Cuda<int>*, size_t);
extern template __global__ void sub_kernel<Cuda<bfloat16>>(Cuda<bfloat16>*, Cuda<bfloat16>*, Cuda<bfloat16>*, size_t);

template<typename DeviceT>
__global__ void sub_scalar_kernel(DeviceT* out, DeviceT* tensor_a, DeviceT scalar, size_t n);

extern template __global__ void sub_scalar_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>*, Cuda<float>, size_t);
extern template __global__ void sub_scalar_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>*, Cuda<int>, size_t);
extern template __global__ void sub_scalar_kernel<Cuda<bfloat16>>(Cuda<bfloat16>*, Cuda<bfloat16>*, Cuda<bfloat16>, size_t);

} // namespace tensor::kernels
