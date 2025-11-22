#pragma once

#include <tensor/tensor.hpp>

namespace llama {
template <tensor::DType T, tensor::Device D> class Embedding {
private:
  tensor::TensorView<T, D> weights_;

public:
  Embedding() = default;
  ~Embedding() = default;

  void set_weights(tensor::TensorView<T, D> weights);

  tensor::Tensor<T, D> forward(tensor::TensorView<int, D> &token_ids) const;
};
} // namespace llama
