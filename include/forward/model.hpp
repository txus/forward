#pragma once

#include <vector>

#include <forward/tokenizer.hpp>

namespace model {
class Model {
private:
  tokenizer::Tokenizer tok;

public:
  explicit Model(std::string_view model_path);
  ~Model();

  std::vector<double> forward(std::vector<int> token_ids);
  std::string generate(std::string_view prompt, uint16_t max_tokens);
};
} // namespace model
