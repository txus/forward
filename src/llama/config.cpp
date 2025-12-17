#include <fstream>
#include <llama/config.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace llama;

ModelConfig llama::load_config(std::string_view model_path) {
  std::ifstream file_stream((std::string(model_path)));

  assert(file_stream.is_open());

  json data = json::parse(file_stream);

  return ModelConfig{
      .vocab_size = data["vocab_size"],
      .head_dim = data["head_dim"],
      .rope_theta = data["rope_theta"],
      .hidden_size = data["hidden_size"],
      .intermediate_size = data["intermediate_size"],
      .max_position_embeddings = data["max_position_embeddings"],
      .num_attention_heads = data["num_attention_heads"],
      .num_hidden_layers = data["num_hidden_layers"],
      .num_key_value_heads = data["num_key_value_heads"],
      .rms_norm_eps = data["rms_norm_eps"],
      .hidden_act = data["hidden_act"],
  };
}
