#pragma once

#include <tensor/tensor.hpp>
#include <variant>

namespace nn {

struct SiLU {
  template <tensor::DType T, tensor::Device D>
  tensor::Tensor<T, D> operator()(tensor::TensorView<T, D> input) const;
};

using Activation = std::variant<SiLU>;

Activation make_activation(std::string_view name);

} // namespace nn
