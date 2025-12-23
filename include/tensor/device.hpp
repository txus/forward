#pragma once

#include <type_traits>

namespace tensor::device {
struct CPU {};

template <typename T> struct is_device : std::false_type {};

template <> struct is_device<CPU> : std::true_type {};

template <typename D> struct device_name;
template <> struct device_name<CPU> {
  static constexpr const char* value = "CPU";
};

#ifdef TENSOR_HAS_CUDA
struct CUDA {};
template <> struct is_device<CUDA> : std::true_type {};
template <> struct device_name<CUDA> {
  static constexpr const char* value = "CUDA";
};
#endif

template <typename D>
concept Device = is_device<D>::value;

} // namespace tensor::device
