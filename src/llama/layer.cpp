#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <llama/layer.hpp>
#include <nn/act.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/ops.hpp>

using namespace llama;
using namespace tensor;
using namespace nn;

template <DType T, Device D>
Layer<T, D>::Layer(const ModelConfig& _config)
    : mlp(MLP<T, D>{_config}), prenorm(RMSNorm<T, D>{_config.rms_norm_eps}),
      postnorm(RMSNorm<T, D>{_config.rms_norm_eps}),
      attention(GroupedQueryAttention<T, D>{_config}) {}

template <DType T, Device D>
void Layer<T, D>::load_weights(std::unordered_map<std::string, Tensor<T, D> /*unused*/>& weight_map,
                               size_t layer_idx) {
  prenorm.set_weights(
      weight_map.at(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx)).view());
  postnorm.set_weights(
      weight_map.at(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx))
          .view());

  attention.load_weights(weight_map, layer_idx);

  mlp.load_weights(weight_map, layer_idx);
}

template <DType T, Device D>
Tensor<T, D> Layer<T, D>::forward(TensorView<T, D> inputs,
                                  const tensor::TensorView<int, D>& attn_mask,
                                  const RoPE<T, D>& rope) const {
  // prenorm
  auto residual_t = prenorm.forward(std::move(inputs));
  auto residual_v = residual_t.view();

  auto attn_output = attention.forward(residual_v, attn_mask, rope);
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
