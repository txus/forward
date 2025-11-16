#pragma once

#include <memory>
#include <string_view>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

namespace tokenizer {
class Tokenizer {
private:
  std::unique_ptr<tokenizers::Tokenizer> impl_;

public:
  explicit Tokenizer(std::string_view tokenizer_path);
  ~Tokenizer();

  std::vector<int> encode(std::string_view prompt);
  std::string decode(std::vector<int> &token_ids);
};
} // namespace tokenizer
