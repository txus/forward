#include <fmt/format.h>

#include <forward/loader.hpp>
#include <fstream>
#include <llama/config.hpp>
#include <llama/model.hpp>
#include <nlohmann/json.hpp>
#include <nn/act.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>
using json = nlohmann::json;

using namespace llama;
using namespace tensor;

ModelConfig load_config(std::string_view model_path) {
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

template <DType T, Device D> Tensor<T, D> llama::causal_attention_mask(size_t seq_len) {
  auto attn_mask = Tensor<T, D>{{seq_len, seq_len}};
  attn_mask.fill_(1);
  return tril(attn_mask.view(), false);
}
template Tensor<int, CPU> llama::causal_attention_mask(size_t seq_len);

template <DType T, Device D>
Model<T, D>::Model(ModelConfig config)
    : config(config), norm(config.rms_norm_eps), rope(config) {} // NOLINT

template <DType T, Device D>
Model<T, D>::Model(std::string_view model_path)
    : config(load_config(model_path)), norm(config.rms_norm_eps), rope(config) {}

template <tensor::DType T, tensor::Device D>
void Model<T, D>::load_weights(
    std::unordered_map<std::string, Tensor<T, D> /*unused*/>& weight_map) {
  embed.set_weights(weight_map.at("model.embed_tokens.weight").view());

  for (int layer_idx = 0; std::cmp_less(layer_idx, config.num_hidden_layers); ++layer_idx) {
    auto layer = Layer<T, D>{config};

    layer.load_weights(weight_map, layer_idx);

    layers.push_back(std::move(layer));
  }

  norm.set_weights(weight_map.at("model.norm.weight").view());

  lm_head.set_weights(weight_map.at("model.embed_tokens.weight").view(), true); // weight tying

  loaded_ = true;
}

template <tensor::DType T, tensor::Device D>
Tensor<T, D> Model<T, D>::forward(TensorView<int, D> token_ids) const {
  assert(loaded_);

  fmt::println("Embedding tokens {}", token_ids);

  auto attn_mask = causal_attention_mask<int, CPU>(token_ids.shape[1]);

  auto residual_stream = embed.forward(token_ids);

  for (int layer_idx = 0; std::cmp_less(layer_idx, config.num_hidden_layers); ++layer_idx) {
    fmt::println("[Layer {}]", layer_idx);
    auto input = residual_stream.view();
    residual_stream = layers[layer_idx].forward(input, attn_mask.view(), rope);
  }

  auto residual_v = residual_stream.view();
  residual_stream = norm.forward(residual_v);

  residual_stream = lm_head.forward(residual_stream.view());

  return residual_stream;
}

template class llama::Model<bfloat16, CPU>;
