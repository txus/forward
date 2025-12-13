#pragma once

#include <llama/config.hpp>
#include <tensor/tensor.hpp>

namespace llama {

template <tensor::DType T, tensor::Device D> class RoPE {
private:
  std::tuple<tensor::Tensor<float, D>, tensor::Tensor<float, D>> cos_sin; // float32

public:
  explicit RoPE(const llama::ModelConfig& config);
  ~RoPE() = default;

  tensor::TensorView<const float, D> cos() const;
  tensor::TensorView<const float, D> sin() const;

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs) const;
};
} // namespace llama
