#include <fstream>
#include <iostream>
#include <string>

#include <forward/tokenizer.hpp>
#include <tokenizers_cpp.h>

std::string LoadBytesFromFile(std::string_view path) {
  std::ifstream fs(std::filesystem::path(path),
                   std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

namespace tokenizer {

Tokenizer::Tokenizer(std::string_view tokenizer_path) {
  auto blob = LoadBytesFromFile(tokenizer_path);
  impl_ = tokenizers::Tokenizer::FromBlobJSON(blob);
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(std::string_view prompt) {
  return impl_->Encode(std::string(prompt));
}
std::string Tokenizer::decode(std::vector<int> token_ids) {
  return impl_->Decode(token_ids);
}

} // namespace tokenizer
