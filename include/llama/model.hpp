#pragma once

#include <llama/config.hpp>
#include <llama/layer.hpp>
#include <nn/embedding.hpp>
#include <nn/linear.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace llama {

template <typename T, typename D> tensor::Tensor<T, D> causal_attention_mask(size_t seq_len);

template <typename T, typename D> class Model {
private:
  size_t kv_cache_size;
  size_t max_tokens;
  bool loaded_ = false;

public:
  explicit Model(ModelConfig config, size_t max_tokens, size_t kv_cache_size = 0);
  explicit Model(std::string_view model_path, size_t max_tokens, size_t kv_cache_size = 0);
  ~Model() = default;

  ModelConfig config;
  nn::Embedding<T, D> embed;
  std::vector<llama::Layer<T, D>> layers{};
  nn::RMSNorm<T, D> norm;
  nn::Linear<T, D> lm_head;
  RoPE<T, D> rope;

  void load_weights(const tensor::Loader<T, D>& loader);

  tensor::Tensor<std::remove_const_t<T>, D> forward(const tensor::TensorView<int, D>& token_ids);
};
} // namespace llama
