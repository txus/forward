#pragma once

#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace nn {

template <typename T, typename D> class RMSNorm {
private:
  tensor::Tensor<const T, D> weights_;
  float eps;

public:
  explicit RMSNorm(float eps);
  ~RMSNorm() = default;
  RMSNorm(RMSNorm&&) noexcept = default;
  RMSNorm& operator=(RMSNorm&&) noexcept = default;
  RMSNorm(const RMSNorm&) = delete;
  RMSNorm& operator=(const RMSNorm&) = delete;

  void load_weights(const tensor::Loader<T, D>& loader, std::string_view name);
  void load_weights(tensor::Tensor<const T, D> weights);
  void load_weights(const tensor::Tensor<T, D>& weights); // borrows from mutable tensor

  tensor::Tensor<T, D> forward(const tensor::TensorView<T, D>& inputs) const;
};
} // namespace nn
