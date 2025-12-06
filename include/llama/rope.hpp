#pragma once

#include <llama/config.hpp>
#include <tensor/tensor.hpp>

namespace llama {

template <tensor::DType T, tensor::Device D> class RoPE {
private:
  std::tuple<tensor::Tensor<T, D>, tensor::Tensor<T, D>> cos_sin;

public:
  explicit RoPE(const llama::ModelConfig& config);
  ~RoPE() = default;

  tensor::TensorView<const T, D> cos() const;
  tensor::TensorView<const T, D> sin() const;

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs) const;
};
} // namespace llama
