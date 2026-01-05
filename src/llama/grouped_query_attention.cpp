#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>
#include <util/nvtx.hpp>

namespace llama {
using namespace tensor;
using namespace nn;

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D>
gqa_forward(const TensorView<T, D>& qs, const TensorView<T, D>& ks,          // NOLINT
            const TensorView<T, D>& vs, const TensorView<int, D>& attn_mask, // NOLINT
            const Softmax& softmax, T scale_factor, size_t group_size, size_t d_out) {
  auto batch_size = qs.shape[0];
  auto queries_len = qs.shape[2];
  auto kvs_len = ks.shape[2];

  // repeat-expand to (batch, [num_kv_groups * group_size], seq_len, head_dim)
  auto keys = repeat_interleave(ks, 1, group_size);
  auto values = repeat_interleave(vs, 1, group_size);

  auto transposed_keys_view = keys.view();
  transposed_keys_view.transpose(2, 3); // (batch, [num_kv_groups*group_size], head_dim, kvs_len)

  // Materialize the transposed keys - cuBLAS requires contiguous tensors
  auto transposed_keys_ = copy(transposed_keys_view);

  Tensor<T, D> attn_scores;
  {
    NVTX_RANGE("attn_scores");
    // scores are (batch, num_heads, queries_len, kvs_len)
    // -- for each query (row), how much does it attend to the key (col)?
    attn_scores = matmul(qs, transposed_keys_.view());
  }

  attn_scores = mul(attn_scores.view(), scale_factor);

  auto attn_mask_reshaped = attn_mask.reshape({1, 1, queries_len, kvs_len});
  attn_scores = masked_fill(attn_scores.view(), attn_mask_reshaped.view(),
                            T(-std::numeric_limits<float>::infinity()));

  auto attn_weights = softmax(attn_scores.view(), -1);

  Tensor<T, D> weighted_values_;
  {
    NVTX_RANGE("weighted_values");
    weighted_values_ = matmul(attn_weights.view(), values.view());
  }

  auto weighted_values_v = weighted_values_.view();

  weighted_values_v.transpose(1, 2);

  return weighted_values_v.reshape({batch_size, queries_len, d_out});
}

template <typename T, typename D>
GroupedQueryAttention<T, D>::GroupedQueryAttention(const ModelConfig& config, size_t cached_tokens)
    : scale(T(1.0F / std::sqrt(static_cast<float>(config.head_dim)))), d_in(config.hidden_size),
      d_out(config.hidden_size), num_heads(config.num_attention_heads), head_dim(d_out / num_heads),
      num_kv_groups(config.num_key_value_heads), group_size(num_heads / num_kv_groups) {
  assert(d_out % num_heads == 0);
  assert(num_heads % num_kv_groups == 0);

  if (cached_tokens > 0) {
    cache.emplace(config, cached_tokens);
  }
}

template <typename T, typename D>
void GroupedQueryAttention<T, D>::load_weights(const Loader<T, D>& loader, size_t layer_idx) {
  q_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.q_proj.weight", layer_idx));
  k_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.k_proj.weight", layer_idx));
  v_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.v_proj.weight", layer_idx));
  out_proj.load_weights(loader, fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
}

template <typename T, typename D> size_t GroupedQueryAttention<T, D>::get_cache_size() {
  if (cache.has_value()) {
    return cache.value().get_current_tokens();
  }
  return 0;
}

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D>
GroupedQueryAttention<T, D>::forward(const TensorView<T, D>& inputs,
                                     std::optional<TensorView<int, D>> attn_mask_,
                                     const RoPE<T, D>& rope) {
  NVTX_RANGE("attention");

  bool use_fused_attn = false;
#if defined(BACKEND_CUDA) && defined(FUSED_ATTN)
  if constexpr (std::is_same_v<D, CUDA>) {
    use_fused_attn = true;
  }
#endif

  size_t batch_size = inputs.shape[0];
  size_t input_seq_len = inputs.shape[1];

  Tensor<T, D> queries;
  Tensor<T, D> keys;
  Tensor<T, D> values;

  TensorView<int, D> attn_mask;
  Tensor<int, D> mask_to_use;
  TensorView<int, D> attention_mask;

  if (!use_fused_attn) {
    attn_mask = attn_mask_.value();
  }

  size_t position_offset = 0;

  if (cache.has_value()) {
    auto& kv_cache = cache.value();

    kv_cache.initialize_if_needed(batch_size);

    // project only the tokens not in the cache
    auto cached_tokens = kv_cache.get_current_tokens();

    // used in RoPE
    position_offset += cached_tokens;

    Tensor<T, D> last_queries;
    Tensor<T, D> last_keys;
    Tensor<T, D> last_values;
    {
      NVTX_RANGE("q_proj");
      last_queries = q_proj.forward(inputs);
    }
    {
      NVTX_RANGE("k_proj");
      last_keys = k_proj.forward(inputs);
    }
    {
      NVTX_RANGE("v_proj");
      last_values = v_proj.forward(inputs);
    }

    auto cached_output = kv_cache.forward(last_keys.view(), last_values.view());

    queries = std::move(last_queries);
    keys = std::move(std::get<0>(cached_output));
    values = std::move(std::get<1>(cached_output));

    if (!use_fused_attn) {
      mask_to_use = slice(attn_mask, 0, cached_tokens, cached_tokens + input_seq_len);
      mask_to_use = slice(mask_to_use.view(), 1, 0, cached_tokens + input_seq_len);
      attention_mask = mask_to_use.view();
    }
  } else {
    {
      NVTX_RANGE("q_proj");
      queries = q_proj.forward(inputs);
    }
    {
      NVTX_RANGE("k_proj");
      keys = k_proj.forward(inputs);
    }
    {
      NVTX_RANGE("v_proj");
      values = v_proj.forward(inputs);
    }
    if (!use_fused_attn) {
      mask_to_use = slice(attn_mask, 0, 0, input_seq_len);
      mask_to_use = slice(mask_to_use.view(), 1, 0, input_seq_len);
      attention_mask = mask_to_use.view();
    }
  }
  auto queries_len = queries.shape()[1];
  auto kvs_len = keys.shape()[1];

  // reshape into heads (batch, seq_len, num_heads, head_dim)
  queries = queries.view().reshape({batch_size, queries_len, num_heads, head_dim});
  keys = keys.view().reshape({batch_size, kvs_len, num_kv_groups, head_dim});
  values = values.view().reshape({batch_size, kvs_len, num_kv_groups, head_dim});

  // transpose qs, ks and vs
  auto queries_v = queries.view();
  auto keys_v = keys.view();
  auto values_v = values.view();

  queries_v.transpose(1, 2); // (batch, num_heads, seq_len, head_dim)
  keys_v.transpose(1, 2);    // (batch, num_kv_groups, seq_len, head_dim)
  values_v.transpose(1, 2);  // (batch, num_kv_groups, seq_len, head_dim)

  queries = rope.forward(queries_v, position_offset);
  queries_v = queries.view();
  keys = rope.forward(keys_v); // (batch, num_heads, seq_len, head_dim)
  keys_v = keys.view();

  Tensor<T, D> attn_out;

  if (use_fused_attn) {
    if constexpr (std::is_same_v<D, CUDA>) {
      attn_out = gqa_forward_fused(queries_v, keys_v, values_v, scale, group_size, d_out, true);
    } else {
      attn_out = gqa_forward(queries_v, keys_v, values_v, attention_mask, softmax, scale,
                             group_size, d_out);
    }
  } else {
    attn_out =
        gqa_forward(queries_v, keys_v, values_v, attention_mask, softmax, scale, group_size, d_out);
  }

  Tensor<T, D> out;
  {
    NVTX_RANGE("o_proj");
    out = out_proj.forward(attn_out.view());
  }

  return out;
}

template class GroupedQueryAttention<bfloat16, CPU>;

#ifdef BACKEND_CUDA
template class GroupedQueryAttention<bfloat16, CUDA>;
#endif
} // namespace llama
