#include <llama/config.hpp>
#include <llama/layer.hpp>
#include <nn/act.hpp>
#include <tensor/ops.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
Layer<T, D>::Layer(const ModelConfig& _config) : mlp(MLP<T, D>{_config}) {}

template <DType T, Device D>
void Layer<T, D>::load_weights(std::unordered_map<std::string, Tensor<T, D> /*unused*/>& weight_map,
                               size_t layer_idx) {
  prenorm.set_weights(
      weight_map.at(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx)).view());
  postnorm.set_weights(
      weight_map.at(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx))
          .view());

  mlp.load_weights(weight_map, layer_idx);
}

template <DType T, Device D> Tensor<T, D> Layer<T, D>::forward(TensorView<T, D> inputs) const {
  // prenorm
  auto residual_t = prenorm.forward(inputs);
  auto residual_v = residual_t.view();

  Tensor<T, D> attn_output{inputs.shape}; // TODO: implement attention
  auto attn_output_v = attn_output.view();

  residual_t = add(attn_output_v, residual_v);
  residual_v = residual_t.view();

  // postnorm
  residual_t = postnorm.forward(residual_v);
  residual_v = residual_t.view();

  // mlp
  residual_t = mlp.forward(residual_v);
  auto mlp_output_v = residual_t.view();

  residual_t = add(mlp_output_v, residual_v);
  residual_v = residual_t.view();

  return residual_t;
}

template class llama::Layer<bfloat16, CPU>;
