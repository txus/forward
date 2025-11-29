#pragma once

#include <nn/rms_norm.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace llama {
template <tensor::DType T, tensor::Device D> class Layer {
private:
  nn::RMSNorm<T, D> prenorm;
  nn::RMSNorm<T, D> postnorm;

public:
  explicit Layer() = default;
  ~Layer() = default;

  void load_weights(std::unordered_map<std::string, tensor::Tensor<T, D>>& weight_map,
                    size_t layer_idx);

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs) const;
};
} // namespace llama
