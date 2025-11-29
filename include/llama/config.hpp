#pragma once

#include <cstddef>
#include <string>

namespace llama {

struct ModelConfig {
  size_t vocab_size;
  size_t hidden_dim;
  size_t intermediate_size;
  size_t num_hidden_layers;
  std::string hidden_act = "silu";
};

} // namespace llama
