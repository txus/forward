#pragma once

#include <llama/rms_norm.hpp>
#include <tensor/tensor.hpp>

namespace llama {
class TransformerLayer {
private:
  llama::RMSNorm rms_norm_1;

public:
  explicit TransformerLayer();
  ~TransformerLayer();

  void load_weights(std::string_view model_path, size_t layer_idx);

  tensor::Tensor<float> forward(tensor::TensorView<float> inputs);
};
} // namespace llama
