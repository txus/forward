#include <nn/act.hpp>

using namespace nn;
using namespace tensor;

Activation nn::make_activation(std::string_view name) {
  if (name == "silu") {
    return SiLU{};
  }
  throw std::runtime_error(fmt::format("Unknown activation: {}", name));
}

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> SiLU::operator()(const TensorView<T, D>& input) const {
  return input.template map<std::remove_const_t<T>>([](T value) {
    std::remove_const_t<T> sigmoid = 1 / (1 + std::exp(-value));
    return value * sigmoid;
  });
}

template Tensor<bfloat16, CPU> SiLU::operator()(const TensorView<const bfloat16, CPU>& input) const;
template Tensor<bfloat16, CPU> SiLU::operator()(const TensorView<bfloat16, CPU>& input) const;
