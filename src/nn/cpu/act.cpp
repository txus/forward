#include <nn/act.hpp>

using namespace nn;
using namespace tensor;

Activation nn::make_activation(std::string_view name) {
  if (name == "silu") {
    return SiLU{};
  }
  throw std::runtime_error(fmt::format("Unknown activation: {}", name));
}

template <DType T, Device D> Tensor<T, D> SiLU::operator()(TensorView<T, D> input) const {
  Tensor<T, D> out{{input.shape}};

  auto in_span = input.span();
  auto out_span = out.span();

  for (size_t i = 0; i < in_span.size(); ++i) {
    auto value = in_span[i];

    T sigmoid = 1 / (1 + std::exp(-value));
    out_span[i] = value * sigmoid;
  }

  return out;
}

template Tensor<bfloat16, CPU> SiLU::operator()(TensorView<bfloat16, CPU> input) const;
