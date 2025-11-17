#pragma once

#include <llama/rms_norm.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace llama {
class Layer {
private:
  llama::RMSNorm rms_norm_1;

public:
  explicit Layer() = default;
  ~Layer() = default;

  void load_weights(
      std::unordered_map<std::string, tensor::Tensor<float>> &weight_map,
      size_t layer_idx);

  tensor::Tensor<float> forward(tensor::TensorView<float> &inputs) const;
};
} // namespace llama
