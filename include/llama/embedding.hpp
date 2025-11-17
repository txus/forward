#pragma once

#include <tensor/tensor.hpp>

namespace llama {
class Embedding {
private:
  tensor::TensorView<float> weights_;

public:
  Embedding() = default;
  ~Embedding() = default;

  void set_weights(tensor::TensorView<float> weights);

  tensor::Tensor<float> forward(tensor::TensorView<int> &token_ids) const;
};
} // namespace llama
