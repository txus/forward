#pragma once

#include <llama/embedding.hpp>
#include <llama/transformer_layer.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace llama {

struct ModelConfig {
  int vocab_size;
  int hidden_dim;
  int num_hidden_layers;
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
  std::vector<llama::Embedding> layers;

  void load_weights(
      std::unordered_map<std::string, tensor::Tensor<float>> &weight_map);

  tensor::Tensor<float> forward(tensor::TensorView<int> &token_ids) const;
};
} // namespace llama
