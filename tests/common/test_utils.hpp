#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <llama/model.hpp>
#include <span>
#include <tensor/tensor.hpp>
#include <utility>

using namespace tensor;

inline Tensor<bfloat16, CPU> make_tensor(const Shape& shape, bfloat16 value) {
  Tensor<tensor::bfloat16, CPU> weights{shape};
  weights.fill_(value);
  return weights;
}

inline std::unordered_map<std::string, tensor::Tensor<tensor::bfloat16, tensor::CPU>>
empty_weights(const llama::ModelConfig& config) {
  std::unordered_map<std::string, tensor::Tensor<tensor::bfloat16, tensor::CPU>> out;

  {
    std::vector<size_t> shape{{config.vocab_size, config.hidden_size}};
    out.insert_or_assign("model.embed_tokens.weight", make_tensor(shape, 0.5));
  }

  std::vector<size_t> shape{config.hidden_size};

  for (int layer_idx = 0; std::cmp_less(layer_idx, config.num_hidden_layers); ++layer_idx) {
    out.insert_or_assign(fmt::format("model.layers.{}.input_layernorm.weight", layer_idx),
                         make_tensor(shape, 0.1));

    out.insert_or_assign(fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx),
                         make_tensor(shape, 0.1));

    out.insert_or_assign(
        fmt::format("model.layers.{}.self_attn.q_proj.weight", layer_idx),
        make_tensor({config.num_attention_heads * config.head_dim, config.hidden_size}, 0.1));
    out.insert_or_assign(
        fmt::format("model.layers.{}.self_attn.k_proj.weight", layer_idx),
        make_tensor({config.num_key_value_heads * config.head_dim, config.hidden_size}, 0.1));
    out.insert_or_assign(
        fmt::format("model.layers.{}.self_attn.v_proj.weight", layer_idx),
        make_tensor({config.num_key_value_heads * config.head_dim, config.hidden_size}, 0.1));
    out.insert_or_assign(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx),
                         make_tensor({config.hidden_size, config.hidden_size}, 0.1));

    out.insert_or_assign(fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx),
                         make_tensor({config.intermediate_size, config.hidden_size}, 0.1));
    out.insert_or_assign(fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx),
                         make_tensor({config.intermediate_size, config.hidden_size}, 0.1));
    out.insert_or_assign(fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx),
                         make_tensor({config.hidden_size, config.intermediate_size}, 0.1));
  }

  out.insert_or_assign("model.norm.weight", make_tensor(shape, 0.1));

  return out;
}

template <typename T>
void tensor_is_close(std::span<const T> tensor_a, std::span<const T> tensor_b,
                     float atol = float(1e-3), float rtol = float(1e-3)) {
  ASSERT_EQ(tensor_a.size(), tensor_b.size()) << "Span sizes differ";

  for (size_t i = 0; i < tensor_a.size(); ++i) {
    T diff = std::fabs(tensor_a[i] - tensor_b[i]);
    T limit = atol + (rtol * std::fabs(tensor_b[i]));
    ASSERT_LE(diff, limit) << "Mismatch at index " << i << ": a=" << tensor_a[i]
                           << " b=" << tensor_b[i] << " diff=" << diff << " limit=" << limit;
  }
}
