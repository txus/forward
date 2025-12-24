#pragma once

#include <tensor/tensor.hpp>
#include <variant>

namespace nn {

struct SiLU {
  template <typename T, typename D>
  tensor::Tensor<std::remove_const_t<T>, D> operator()(const tensor::TensorView<T, D>& input) const;
};

using Activation = std::variant<SiLU>;

Activation make_activation(std::string_view name);

} // namespace nn
