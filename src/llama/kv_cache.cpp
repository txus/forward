#include <llama/config.hpp>
#include <llama/kv_cache.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
KVCache<T, D>::KVCache(const ModelConfig& _config, size_t max_tokens)
    : max_tokens(max_tokens), num_heads(_config.num_key_value_heads), head_dim(_config.head_dim) {}

template <DType T, Device D>
std::tuple<tensor::Tensor<T, D>, tensor::Tensor<T, D>>
KVCache<T, D>::forward(tensor::TensorView<T, D> new_keys, tensor::TensorView<T, D> new_values) {
  auto& k_cache = get_k_cache();
  auto& v_cache = get_v_cache();

  auto new_tokens_count = new_keys.shape[1];
  assert(new_values.shape[1] == new_tokens_count &&
         "new keys and values have different seq lengths");

  tensor::Tensor<T, D> all_keys;
  tensor::Tensor<T, D> all_values;

  if (current_tokens > 0) { // decode
    auto already_cached_keys = slice(k_cache.view(), 1, 0, current_tokens);
    auto already_cached_values = slice(v_cache.view(), 1, 0, current_tokens);

    all_keys = cat(already_cached_keys.view(), new_keys, 1);
    all_values = cat(already_cached_values.view(), new_values, 1);
  } else { // prefill
    all_keys = new_keys.copy();
    all_values = new_values.copy();
  }

  k_cache.replace_from_(all_keys.view());
  v_cache.replace_from_(all_values.view());

  current_tokens += new_tokens_count;

  return std::tuple(all_keys, all_values);
}

template class llama::KVCache<bfloat16, CPU>;
