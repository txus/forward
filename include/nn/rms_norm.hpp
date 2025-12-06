#pragma once

#include <tensor/tensor.hpp>

namespace nn {

template <tensor::DType T, tensor::Device D> class RMSNorm {
private:
  tensor::TensorView<T, D> weights_;
  float eps = 1e-05;

public:
  explicit RMSNorm(float eps);
  explicit RMSNorm() = default;
  ~RMSNorm() = default;

  void set_weights(tensor::TensorView<T, D> weights);

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs) const;
};
} // namespace nn
