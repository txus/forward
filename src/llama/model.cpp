#include <forward/loader.hpp>
#include <llama/embedding.hpp>
#include <llama/model.hpp>
#include <tensor/tensor.hpp>

namespace llama {

Model::Model(ModelConfig config) : config(config) {}

void Model::load_weights(
    std::unordered_map<std::string_view, tensor::Tensor<float>> &weight_map) {
  embed.set_weights(weight_map.at("model.embed_tokens.weight").view());
  loaded_ = true;
}

tensor::Tensor<float> Model::forward(tensor::TensorView<int> &token_ids) const {
  assert(loaded_);

  auto embedded = embed.forward(token_ids);

  return embedded;
}
} // namespace llama
