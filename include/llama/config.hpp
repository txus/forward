#pragma once

#include <cstddef>
#include <string>

namespace llama {

struct ModelConfig {
  size_t vocab_size = 128256;
  size_t head_dim = 64;
  float rope_theta = 500000.0;
  size_t hidden_size = 2048;
  size_t intermediate_size = 8192;
  size_t max_position_embeddings = 131072;
  size_t num_attention_heads = 32;
  size_t num_hidden_layers = 16;
  size_t num_key_value_heads = 8;
  float rms_norm_eps = 1e-05;
  std::string hidden_act = "silu";
};

ModelConfig load_config(std::string_view model_path);

} // namespace llama
