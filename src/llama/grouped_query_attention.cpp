#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
GroupedQueryAttention<T, D>::GroupedQueryAttention(const ModelConfig& config)
    : scale(T(1.0F / std::sqrt(static_cast<float>(config.head_dim)))), d_in(config.hidden_size),
      d_out(config.hidden_size), num_heads(config.num_attention_heads), head_dim(d_out / num_heads),
      num_kv_groups(config.num_key_value_heads), group_size(num_heads / num_kv_groups) {
  assert(d_out % num_heads == 0);
  assert(num_heads % num_kv_groups == 0);
}

template <DType T, Device D>
void GroupedQueryAttention<T, D>::load_weights(const tensor::Loader<T, D>& loader,
                                               size_t layer_idx) {
  q_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.q_proj.weight", layer_idx));
  k_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.k_proj.weight", layer_idx));
  v_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.v_proj.weight", layer_idx));
  out_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
}

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D>
GroupedQueryAttention<T, D>::forward(const TensorView<T, D>& inputs,
                                     const TensorView<int, D>& attn_mask, const RoPE<T, D>& rope) {
  size_t batch_size = inputs.shape[0];
  size_t seq_len = inputs.shape[1];

  // project queries, keys and values: (batch, seq_len, (num_heads (or num_kv_groups))*head_dim)
  auto queries = q_proj.forward(inputs);
  auto keys = k_proj.forward(inputs);
  auto values = v_proj.forward(inputs);

  // reshape into heads (batch, seq_len, num_heads, head_dim)
  queries = queries.view().reshape({batch_size, seq_len, num_heads, head_dim});
  keys = keys.view().reshape({batch_size, seq_len, num_kv_groups, head_dim});
  values = values.view().reshape({batch_size, seq_len, num_kv_groups, head_dim});

  // transpose qs ,ks and vs
  auto queries_v = queries.view();
  auto keys_v = keys.view();
  auto values_v = values.view();

  queries_v.transpose(1, 2); // (batch, num_heads, seq_len, head_dim)
  keys_v.transpose(1, 2);    // (batch, num_kv_groups, seq_len, head_dim)
  values_v.transpose(1, 2);  // (batch, num_kv_groups, seq_len, head_dim)

  queries = rope.forward(queries_v);
  queries_v = queries.view();
  keys = rope.forward(keys_v); // (batch, num_heads, seq_len, head_dim)
  keys_v = keys.view();

  // repeat-expand to (batch, [num_kv_groups * group_size], seq_len, head_dim)
  keys = keys_v.repeat_interleave(1, group_size);
  values = values_v.repeat_interleave(1, group_size);


  auto transposed_keys_ = keys.view();
  transposed_keys_.transpose(2, 3); // (batch, [num_kv_groups*group_size], head_dim, seq_len)

  // scores are (batch, num_heads, seq_len, seq_len)
  // -- for each query (row), how much does it attend to the key (col)?
  auto attn_scores = matmul(queries_v, transposed_keys_);

  attn_scores = mul(attn_scores.view(), scale);

  auto attn_mask_reshaped = attn_mask.reshape({1, 1, seq_len, seq_len});
  attn_scores = masked_fill(attn_scores.view(), attn_mask_reshaped.view(),
                            T(-std::numeric_limits<float>::infinity()));

  auto attn_weights = softmax(attn_scores.view(), -1);

  auto weighted_values_ = matmul(attn_weights.view(), values.view());

  auto weighted_values_v = weighted_values_.view();

  weighted_values_v.transpose(1, 2);

  auto materialized_weighted_values = weighted_values_v.reshape({batch_size, seq_len, d_out});

  auto out = out_proj.forward(materialized_weighted_values.view());

  return out;
}

template class llama::GroupedQueryAttention<bfloat16, CPU>;
