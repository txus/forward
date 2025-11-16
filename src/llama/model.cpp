#include <fmt/format.h>
#include <forward/loader.hpp>
#include <llama/embedding.hpp>
#include <llama/model.hpp>
#include <tensor/tensor.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace llama {

Model::Model(ModelConfig config) : config(config) {}
Model::Model(std::string_view model_path) {
  std::ifstream f((std::string(model_path)));

  assert(f.is_open());

  json data = json::parse(f);

  config = ModelConfig{.vocab_size = data["vocab_size"],
                       .hidden_dim = data["hidden_size"],
                       .num_hidden_layers = data["num_hidden_layers"]};
}

void Model::load_weights(
    std::unordered_map<std::string, tensor::Tensor<float>> &weight_map) {
  auto embed_weights = weight_map.at("model.embed_tokens.weight").view();
  fmt::println("loading embedding weights {}", embed_weights);
  embed.set_weights(embed_weights);
  loaded_ = true;
}

tensor::Tensor<float> Model::forward(tensor::TensorView<int> &token_ids) const {
  assert(loaded_ == true);

  fmt::println("Embedding tokens {}", token_ids);

  auto embedded = embed.forward(token_ids);

  return embedded;
}
} // namespace llama
