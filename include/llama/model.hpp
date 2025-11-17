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

class Model {
private:
  bool loaded_ = false;

public:
  explicit Model(ModelConfig config);
  explicit Model(std::string_view model_path);
  ~Model() = default;

  ModelConfig config;
  llama::Embedding embed;
  std::vector<llama::Layer> layers;

  void load_weights(
      std::unordered_map<std::string, tensor::Tensor<float>> &weight_map);

  tensor::Tensor<float> forward(tensor::TensorView<int> &token_ids) const;
};
} // namespace llama
