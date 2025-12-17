#pragma once

#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <llama/mlp.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace llama {
template <tensor::DType T, tensor::Device D> class Layer {
private:
  nn::RMSNorm<T, D> prenorm;
  nn::RMSNorm<T, D> postnorm;
  MLP<T, D> mlp;
  GroupedQueryAttention<T, D> attention;

public:
  explicit Layer(const ModelConfig& config);
  ~Layer() = default;

  void load_weights(const tensor::Loader<T, D>& loader, size_t layer_idx);

  tensor::Tensor<std::remove_const_t<T>, D> forward(const tensor::TensorView<T, D>& inputs,
                                                    const tensor::TensorView<int, D>& attn_mask,
                                                    const RoPE<T, D>& rope);
};
} // namespace llama
