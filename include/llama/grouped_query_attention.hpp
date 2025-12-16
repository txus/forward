#pragma once

#include <llama/rope.hpp>
#include <nn/linear.hpp>
#include <nn/softmax.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace llama {

template <tensor::DType T, tensor::Device D> class GroupedQueryAttention {
private:
  size_t d_in;
  size_t d_out;
  size_t num_heads;
  size_t head_dim;
  size_t num_kv_groups;
  size_t group_size;

  nn::Linear<T, D> q_proj;
  nn::Linear<T, D> k_proj;
  nn::Linear<T, D> v_proj;
  nn::Linear<T, D> out_proj;

  nn::Softmax softmax;

public:
  explicit GroupedQueryAttention(const ModelConfig& config);
  ~GroupedQueryAttention() = default;

  void load_weights(std::unordered_map<std::string, tensor::Tensor<T, D>>& weight_map,
                    size_t layer_idx);

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs,
                               const tensor::TensorView<int, D>& attn_mask,
                               const RoPE<T, D>& rope) const;
};
} // namespace llama
