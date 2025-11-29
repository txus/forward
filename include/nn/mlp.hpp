#pragma once

#include <tensor/tensor.hpp>

namespace nn {
template <tensor::DType T, tensor::Device D> class MLP {
private:
  tensor::TensorView<T, D> weights_;

public:
  MLP() = default;
  ~MLP() = default;

  void set_weights(tensor::TensorView<T, D> weights);

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs) const;
};
} // namespace nn
