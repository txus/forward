#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
GroupedQueryAttention<T, D>::GroupedQueryAttention(const ModelConfig& config)
    : d_in(config.hidden_size), d_out(config.hidden_size), num_heads(config.num_attention_heads),
      head_dim(d_out / num_heads), num_kv_groups(config.num_key_value_heads),
      group_size(num_heads / num_kv_groups) {
  assert(d_out % num_heads == 0);
  assert(num_heads % num_kv_groups == 0);
}

template <DType T, Device D>
void GroupedQueryAttention<T, D>::load_weights(const tensor::Loader<T, D>& loader,
                                               size_t layer_idx) {
  q_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.q_proj.weight", layer_idx),
                      true);
  k_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.k_proj.weight", layer_idx),
                      true);
  v_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.v_proj.weight", layer_idx),
                      true);
  out_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx),
                        true);
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
  auto queries_v = queries.view().view_as({batch_size, seq_len, num_heads, head_dim});
  auto keys_v = keys.view().view_as({batch_size, seq_len, num_kv_groups, head_dim});
  auto values_v = values.view().view_as({batch_size, seq_len, num_kv_groups, head_dim});

  // transpose qs ,ks and vs
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
  auto transposed_keys = transposed_keys_.copy();

  // scores are (batch, num_heads, seq_len, seq_len)
  // -- for each query (row), how much does it attend to the key (col)?
  auto attn_scores = matmul(queries_v, transposed_keys.view());

  // apply causal mask
  attn_scores = masked_fill(attn_scores.view(), attn_mask.view_as({1, 1, seq_len, seq_len}),
                            T(-std::numeric_limits<T>::infinity()));

  attn_scores = div(attn_scores.view(), T(std::pow(head_dim, 0.5)));

  auto attn_weights = softmax(attn_scores.view(), -1);

  auto weighted_values_ = matmul(attn_weights.view(), values.view());

  auto weighted_values_v = weighted_values_.view();

  weighted_values_v.transpose(1, 2);

  auto copied = weighted_values_v.copy();

  auto viewed = copied.view();

  auto reshaped_ = viewed.view_as({batch_size, seq_len, d_out});

  auto reshaped = reshaped_.copy();

  auto out = out_proj.forward(reshaped.view());

  return out;
}

template class llama::GroupedQueryAttention<bfloat16, CPU>;
