#pragma once

#include <llama/config.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>
#include <tuple>

namespace llama {
template <typename T, typename D> class KVCache {
private:
  std::optional<tensor::Tensor<T, D>> k_cache;
  std::optional<tensor::Tensor<T, D>> v_cache;

  size_t num_heads;
  size_t head_dim;

  size_t max_tokens;
  size_t current_tokens = 0;

public:
  explicit KVCache(const ModelConfig& config, size_t max_tokens);
  ~KVCache() = default;
  KVCache(KVCache&&) noexcept = default;
  KVCache& operator=(KVCache&&) noexcept = default;
  KVCache(const KVCache&) = delete;
  KVCache& operator=(const KVCache&) = delete;

  void initialize_if_needed(size_t batch_size) {
    if (!k_cache.has_value()) {
      k_cache.emplace({batch_size, max_tokens, num_heads * head_dim});
      v_cache.emplace({batch_size, max_tokens, num_heads * head_dim});
    }
  }

  std::tuple<tensor::Tensor<T, D>, tensor::Tensor<T, D>>
  forward(tensor::TensorView<T, D> new_keys, tensor::TensorView<T, D> new_values);

  auto& get_current_tokens() {
    return current_tokens;
  }

  auto& get_k_cache() {
    return k_cache.value();
  }

  auto& get_v_cache() {
    return v_cache.value();
  }
};
} // namespace llama
