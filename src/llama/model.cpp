#include <fmt/format.h>
#include <forward/loader.hpp>
#include <llama/model.hpp>
#include <tensor/tensor.hpp>

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace llama;
using namespace tensor;

template <DType T, Device D>
Model<T, D>::Model(ModelConfig config) : config(config) {}
template <DType T, Device D> Model<T, D>::Model(std::string_view model_path) {
  std::ifstream f((std::string(model_path)));

  assert(f.is_open());

  json data = json::parse(f);

  config = ModelConfig{.vocab_size = data["vocab_size"],
                       .hidden_dim = data["hidden_size"],
                       .num_hidden_layers = data["num_hidden_layers"]};
}

template <DType T, Device D>
void Model<T, D>::load_weights(
    std::unordered_map<std::string, Tensor<T, D>> &weight_map) {
  embed.set_weights(weight_map.at("model.embed_tokens.weight").view());

  for (int l = 0; l < config.num_hidden_layers; ++l) {
    auto layer = Layer<T, D>{};

    layer.load_weights(weight_map, l);

    layers.push_back(std::move(layer));
  }

  loaded_ = true;
}

template <DType T, Device D>
Tensor<T, D> Model<T, D>::forward(TensorView<int, D> &token_ids) const {
  assert(loaded_ == true);

  fmt::println("Embedding tokens {}", token_ids);

  auto residual_stream = embed.forward(token_ids);

  for (int l = 0; l < config.num_hidden_layers; ++l) {
    auto input = residual_stream.view();
    residual_stream = layers[l].forward(input);
  }

  return residual_stream;
}

template class llama::Model<bfloat16, CPU>;
