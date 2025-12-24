#pragma once

#include <tensor/device.hpp>
#include <tensor/dtype.hpp>

#ifdef TENSOR_HAS_CUDA
#include <cuda_bf16.h>
#endif

namespace tensor {

using namespace device;
using namespace dtype;

// Note: Using typename instead of DType/Device concepts to avoid ABI mismatch

// Primary template: by default, use the same type on all devices
template <typename T, typename D>
struct device_type {
  using type = T;
};

// Helper alias for convenience
template <typename T, typename D>
using device_type_t = typename device_type<T, D>::type;

#ifdef TENSOR_HAS_CUDA
// Short alias for CUDA device types: Cuda<float> -> float, Cuda<bfloat16> -> __nv_bfloat16
template <typename T>
using Cuda = device_type_t<T, CUDA>;
#endif

#ifdef TENSOR_HAS_CUDA
// CUDA specialization: bfloat16 -> __nv_bfloat16
template <>
struct device_type<bfloat16, CUDA> {
  using type = __nv_bfloat16;
};

// Conversion utilities for bfloat16 <-> __nv_bfloat16
__host__ __device__ inline __nv_bfloat16 to_device_type(bfloat16 val, CUDA) {
  return __nv_bfloat16_raw{val.bits};
}

__host__ __device__ inline bfloat16 from_device_type(__nv_bfloat16 val, CUDA) {
  bfloat16 result;
  result.bits = __nv_bfloat16_raw(val).x;
  return result;
}

// Pass-through for types that don't need conversion
template <typename T>
__host__ __device__ inline T to_device_type(T val, CUDA) {
  return val;
}

template <typename T>
__host__ __device__ inline T from_device_type(T val, CUDA) {
  return val;
}
#endif

// CPU: all types pass through unchanged
template <typename T>
inline T to_device_type(T val, CPU) {
  return val;
}

template <typename T>
inline T from_device_type(T val, CPU) {
  return val;
}

// Check if a type needs conversion for a given device
template <typename T, typename D>
struct needs_device_conversion : std::bool_constant<!std::is_same_v<T, device_type_t<T, D>>> {};

template <typename T, typename D>
inline constexpr bool needs_device_conversion_v = needs_device_conversion<T, D>::value;

} // namespace tensor
