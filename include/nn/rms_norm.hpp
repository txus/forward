#pragma once

#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace nn {

template <tensor::DType T, tensor::Device D> class RMSNorm {
private:
  tensor::TensorView<const T, D> weights_;
  float eps;

public:
  explicit RMSNorm(float eps);
  ~RMSNorm() = default;

  void load_weights(const tensor::Loader<T, D>& loader, std::string_view name);
  void load_weights(tensor::TensorView<const T, D> weights);
  void load_weights(tensor::TensorView<T, D> weights);

  tensor::Tensor<std::remove_const_t<T>, D> forward(const tensor::TensorView<T, D>& inputs) const;
};
} // namespace nn
