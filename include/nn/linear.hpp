#pragma once

#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace nn {
template <tensor::DType T, tensor::Device D> class Linear {
private:
  tensor::Tensor<const T, D> weights_;
  tensor::TensorView<const T, D> weights_t_; // transposed view for matmul

public:
  Linear() = default;
  ~Linear() = default;

  void load_weights(const tensor::Loader<T, D>& loader, std::string_view name);

  tensor::Tensor<T, D> forward(const tensor::TensorView<T, D>& inputs);
};
} // namespace nn
