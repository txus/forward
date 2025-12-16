#pragma once

#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <llama/mlp.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

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

  void load_weights(std::unordered_map<std::string, tensor::Tensor<T, D>>& weight_map,
                    size_t layer_idx);

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs,
                               const tensor::TensorView<int, D>& attn_mask,
                               const RoPE<T, D>& rope) const;
};
} // namespace llama
