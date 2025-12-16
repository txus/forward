#pragma once

#include <llama/config.hpp>
#include <llama/layer.hpp>
#include <nn/embedding.hpp>
#include <nn/linear.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace llama {

template <tensor::DType T, tensor::Device D>
tensor::Tensor<T, D> causal_attention_mask(size_t seq_len);

template <tensor::DType T, tensor::Device D> class Model {
private:
  bool loaded_ = false;

public:
  explicit Model(ModelConfig config);
  explicit Model(std::string_view model_path);
  ~Model() = default;

  ModelConfig config;
  nn::Embedding<T, D> embed;
  std::vector<llama::Layer<T, D>> layers{};
  nn::RMSNorm<T, D> norm;
  nn::Linear<T, D> lm_head;
  RoPE<T, D> rope;

  void load_weights(std::unordered_map<std::string, tensor::Tensor<T, D>>& weight_map);

  tensor::Tensor<T, D> forward(tensor::TensorView<int, D> token_ids) const;
};
} // namespace llama
