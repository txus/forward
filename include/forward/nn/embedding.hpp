#pragma once

#include <forward/tensor.hpp>

namespace nn {
class Embedding {
private:
  tensor::Tensor<float> weights;

public:
  explicit Embedding(tensor::Tensor<float> weights);
  ~Embedding();

  tensor::Tensor<float> forward(tensor::Tensor<int> token_ids);
};
} // namespace nn
