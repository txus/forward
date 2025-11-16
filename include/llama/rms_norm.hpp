#pragma once

#include <tensor/tensor.hpp>

namespace llama {
class RMSNorm {
private:
  tensor::TensorView<float> weights;

public:
  explicit RMSNorm();
  ~RMSNorm();

  void load_weights(std::string_view model_path, std::string_view tensor_name);

  tensor::Tensor<float> forward(tensor::TensorView<float> inputs);
};
} // namespace llama
