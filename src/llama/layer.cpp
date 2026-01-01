#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <llama/layer.hpp>
#include <nn/act.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/ops.hpp>

using namespace llama;
using namespace tensor;
using namespace nn;

template <typename T, typename D>
Layer<T, D>::Layer(const ModelConfig& _config, size_t cached_tokens)
    : mlp(MLP<T, D>{_config}), prenorm(RMSNorm<T, D>{_config.rms_norm_eps}),
      postnorm(RMSNorm<T, D>{_config.rms_norm_eps}),
      attention(GroupedQueryAttention<T, D>{_config, cached_tokens}) {}

template <typename T, typename D>
void Layer<T, D>::load_weights(const tensor::Loader<T, D>& loader, size_t layer_idx) {
  prenorm.load_weights(loader, fmt::format("model.layers.{}.input_layernorm.weight", layer_idx));
  postnorm.load_weights(loader,
                        fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx));

  attention.load_weights(loader, layer_idx);

  mlp.load_weights(loader, layer_idx);
}

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> Layer<T, D>::forward(const TensorView<T, D>& inputs,
                                                       const TensorView<int, D>& attn_mask,
                                                       const RoPE<T, D>& rope) {
  auto residual = inputs;

  // prenorm
  auto normalized = prenorm.forward(std::move(inputs));

  auto attn_output = attention.forward(normalized.view(), attn_mask, rope);

  auto residual_ = add(attn_output.view(), residual);
  residual = residual_.view();

  // postnorm
  normalized = postnorm.forward(residual);

  // mlp
  auto mlp_out = mlp.forward(normalized.view());

  residual_ = add(mlp_out.view(), residual);

  return residual_;
}

template class llama::Layer<bfloat16, CPU>;

#ifdef BACKEND_CUDA
template class llama::Layer<bfloat16, CUDA>;
#endif
