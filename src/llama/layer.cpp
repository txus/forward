#include <llama/layer.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
void Layer<T, D>::load_weights(std::unordered_map<std::string, Tensor<T, D> /*unused*/>& weight_map,
                               size_t layer_idx) {

  const std::string key = fmt::format("model.layers.{}.input_layernorm.weight", layer_idx);

  rms_norm_1.set_weights(weight_map.at(key).view());
}

template <DType T, Device D> Tensor<T, D> Layer<T, D>::forward(TensorView<T, D>& inputs) const {
  auto norm = rms_norm_1.forward(inputs);

  return norm;
}

template class llama::Layer<bfloat16, CPU>;
