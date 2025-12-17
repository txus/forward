#pragma once

#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace nn {
template <tensor::DType T, tensor::Device D> class Embedding {
private:
  tensor::TensorView<const T, D> weights_;

public:
  Embedding() = default;
  ~Embedding() = default;

  void load_weights(const tensor::Loader<T, D>& loader);
  void load_weights(tensor::TensorView<const T, D> weights);
  void load_weights(tensor::TensorView<T, D> weights);

  tensor::Tensor<std::remove_const_t<T>, D>
  forward(const tensor::TensorView<int, D>& token_ids) const;
};
} // namespace nn
