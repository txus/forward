#pragma once

#include <type_traits>

namespace tensor {
namespace device {
struct CPU {};
struct CUDA {};

template <typename T> struct is_device : std::false_type {};

template <> struct is_device<CPU> : std::true_type {};
template <> struct is_device<CUDA> : std::true_type {};

template <typename D>
concept Device = is_device<D>::value;

template <typename D> struct device_name;
template <> struct device_name<CPU> {
  static constexpr const char *value = "CPU";
};
template <> struct device_name<CUDA> {
  static constexpr const char *value = "CUDA";
};
} // namespace device
} // namespace tensor
