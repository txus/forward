#include <tokenizers_cpp.h>

#include <forward/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <string>

std::string LoadBytesFromFile(std::string_view path) {
  std::ifstream file_stream(std::filesystem::path(path), std::ios::in | std::ios::binary);
  if (file_stream.fail()) {
    std::cerr << "Cannot open " << path << '\n';
    exit(1);
  }
  std::string data;
  file_stream.seekg(0, std::ios::end);
  long size = static_cast<long>(file_stream.tellg());
  file_stream.seekg(0, std::ios::beg);
  data.resize(size);
  file_stream.read(data.data(), size);
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
std::string Tokenizer::decode(std::vector<int>& token_ids) {
  return impl_->Decode(token_ids);
}

} // namespace tokenizer
