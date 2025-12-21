#include <fmt/format.h>

#include <llama/config.hpp>
#include <llama/model.hpp>
#include <nlohmann/json.hpp>
#include <nn/act.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/loader.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>
using json = nlohmann::json;

using namespace llama;
using namespace tensor;

template <DType T, Device D> Tensor<T, D> llama::causal_attention_mask(size_t seq_len) {
  auto attn_mask = Tensor<T, D>{{seq_len, seq_len}};
  attn_mask.fill_(1);
  return tril(attn_mask.view(), false);
}
template Tensor<int, CPU> llama::causal_attention_mask(size_t seq_len);

template <DType T, Device D>
Model<T, D>::Model(ModelConfig config, size_t max_tokens, size_t kv_cache_size)
    : max_tokens(max_tokens), kv_cache_size(kv_cache_size), config(config),
      norm(config.rms_norm_eps), rope(config) {} // NOLINT

template <DType T, Device D>
Model<T, D>::Model(std::string_view model_path, size_t max_tokens, size_t kv_cache_size)
    : max_tokens(max_tokens), kv_cache_size(kv_cache_size), config(load_config(model_path)),
      norm(config.rms_norm_eps), rope(config) {}

template <tensor::DType T, tensor::Device D>
void Model<T, D>::load_weights(const tensor::Loader<T, D>& loader) {
  embed.load_weights(loader);

  for (int layer_idx = 0; std::cmp_less(layer_idx, config.num_hidden_layers); ++layer_idx) {
    auto layer = Layer<T, D>{config, kv_cache_size};

    layer.load_weights(loader, layer_idx);

    layers.push_back(std::move(layer));
  }

  norm.load_weights(loader, "model.norm.weight");

  lm_head.load_weights(loader, "model.embed_tokens.weight");

  loaded_ = true;
}

template <tensor::DType T, tensor::Device D>
Tensor<std::remove_const_t<T>, D> Model<T, D>::forward(const TensorView<int, D>& token_ids) {
  assert(loaded_);

  fmt::println("Embedding tokens {}", token_ids);

  auto attn_mask = causal_attention_mask<int, CPU>(max_tokens);

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
