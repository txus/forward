#pragma once

#include <tensor/tensor.hpp>

namespace llama {
class RMSNorm {
private:
  tensor::TensorView<float> weights_;

public:
  explicit RMSNorm() = default;
  ~RMSNorm() = default;

  void set_weights(tensor::TensorView<float> weights);

  tensor::Tensor<float> forward(tensor::TensorView<float> &inputs) const;
};
} // namespace llama
