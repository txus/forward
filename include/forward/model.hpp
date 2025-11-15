#pragma once

#include <forward/nn/embedding.hpp>
#include <forward/tensor.hpp>
#include <forward/tokenizer.hpp>

namespace model {
class Model {
private:
  tokenizer::Tokenizer tok;
  nn::Embedding embed;

public:
  explicit Model(std::string_view model_path);
  ~Model();

  tensor::Tensor<float> forward(tensor::Tensor<int> token_ids);
  std::string generate(std::string_view prompt, uint16_t max_tokens);
};
} // namespace model
