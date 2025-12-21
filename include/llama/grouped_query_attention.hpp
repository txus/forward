#pragma once

#include <llama/kv_cache.hpp>
#include <llama/rope.hpp>
#include <nn/linear.hpp>
#include <nn/softmax.hpp>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace llama {

template <tensor::DType T, tensor::Device D> class GroupedQueryAttention {
private:
  size_t d_in;
  size_t d_out;
  size_t num_heads;
  size_t head_dim;
  size_t num_kv_groups;
  size_t group_size;

  T scale;

  nn::Linear<T, D> q_proj;
  nn::Linear<T, D> k_proj;
  nn::Linear<T, D> v_proj;
  nn::Linear<T, D> out_proj;

  nn::Softmax softmax;

  std::optional<KVCache<T, D>> cache;

public:
  explicit GroupedQueryAttention(const ModelConfig& config, size_t cached_tokens = 0);
  ~GroupedQueryAttention() = default;

  void load_weights(const tensor::Loader<T, D>& loader, size_t layer_idx);

  tensor::Tensor<std::remove_const_t<T>, D> forward(const tensor::TensorView<T, D>& inputs,
                                                    const tensor::TensorView<int, D>& attn_mask,
                                                    const RoPE<T, D>& rope);

  size_t get_cache_size();
};
} // namespace llama
