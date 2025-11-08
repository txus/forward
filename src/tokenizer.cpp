#include <fstream>
#include <iostream>
#include <string>

#include "forward/tokenizer.hpp"

#include <tokenizers_cpp.h>

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
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

std::vector<int> encode(std::string prompt) {
  // Read blob from file.
  auto blob = LoadBytesFromFile("model/tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  // call Encode to turn prompt into token ids
  std::vector<int> ids = tok->Encode(prompt);
  // call Decode to turn ids into string
  std::string decoded_prompt = tok->Decode(ids);
  return ids;
}

std::string decode(std::vector<int> token_ids) {
  // Read blob from file.
  auto blob = LoadBytesFromFile("model/tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
  // call Decode to turn ids into string
  std::string decoded_prompt = tok->Decode(token_ids);

  return decoded_prompt;
}
