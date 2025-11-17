#include <llama/layer.hpp>

using namespace llama;

void Layer::load_weights(
    std::unordered_map<std::string, tensor::Tensor<float>> &weight_map,
    size_t layer_idx) {

  const std::string key =
      fmt::format("model.layers.{}.input_layernorm.weight", layer_idx);

  rms_norm_1.set_weights(weight_map.at(key).view());
}

tensor::Tensor<float> Layer::forward(tensor::TensorView<float> &inputs) const {
  auto norm = rms_norm_1.forward(inputs);

  return norm;
}
