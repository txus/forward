#pragma once

#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace nn {
template <tensor::DType T, tensor::Device D> class Embedding {
private:
  tensor::Tensor<const T, D> weights_;

public:
  Embedding() = default;
  ~Embedding() = default;

  void load_weights(const tensor::Loader<T, D>& loader);
  void load_weights(tensor::Tensor<const T, D> weights);
  void load_weights(const tensor::Tensor<T, D>& weights); // borrows from mutable tensor

  tensor::Tensor<T, D> forward(const tensor::TensorView<int, D>& token_ids) const;
};
} // namespace nn
