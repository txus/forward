#pragma once

#include <llama/embedding.hpp>
#include <llama/layer.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace llama {

struct ModelConfig {
  size_t vocab_size;
  size_t hidden_dim;
  size_t num_hidden_layers;
};

template <tensor::DType T, tensor::Device D> class Model {
private:
  bool loaded_ = false;

public:
  explicit Model(ModelConfig config);
  explicit Model(std::string_view model_path);
  ~Model() = default;

  ModelConfig config;
  llama::Embedding<T, D> embed;
  std::vector<llama::Layer<T, D>> layers;

  void load_weights(
      std::unordered_map<std::string, tensor::Tensor<T, D>> &weight_map);

  tensor::Tensor<T, D> forward(tensor::TensorView<int, D> &token_ids) const;
};
} // namespace llama
