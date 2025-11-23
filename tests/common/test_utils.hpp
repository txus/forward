#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <llama/model.hpp>
#include <span>
#include <tensor/tensor.hpp>
#include <utility>

inline std::unordered_map<std::string, tensor::Tensor<tensor::bfloat16, tensor::CPU>>
empty_weights(const llama::ModelConfig& config) {
  std::unordered_map<std::string, tensor::Tensor<tensor::bfloat16, tensor::CPU>> out;

  {
    std::vector<size_t> shape{{config.vocab_size, config.hidden_dim}};

    tensor::Tensor<tensor::bfloat16, tensor::CPU> weights{shape};
    weights.fill_(tensor::bfloat16(0.5));

    out.insert_or_assign("model.embed_tokens.weight", std::move(weights));
  }

  for (int layer_idx = 0; std::cmp_less(layer_idx, config.num_hidden_layers); ++layer_idx) {
    std::vector<size_t> shape{config.hidden_dim};

    const std::string prenorm_key =
        fmt::format("model.layers.{}.input_layernorm.weight", layer_idx);

    tensor::Tensor<tensor::bfloat16, tensor::CPU> prenorm_weights{shape};
    prenorm_weights.fill_(tensor::bfloat16(0.1));

    out.insert_or_assign(prenorm_key, std::move(prenorm_weights));

    const std::string postnorm_key =
        fmt::format("model.layers.{}.post_attention_layernorm.weight", layer_idx);

    tensor::Tensor<tensor::bfloat16, tensor::CPU> postnorm_weights{shape};
    postnorm_weights.fill_(tensor::bfloat16(0.1));

    out.insert_or_assign(postnorm_key, std::move(prenorm_weights));
  }

  std::vector<size_t> shape{config.hidden_dim};

  tensor::Tensor<tensor::bfloat16, tensor::CPU> norm_weights{shape};
  norm_weights.fill_(tensor::bfloat16(0.1));

  out.insert_or_assign("model.norm.weight", std::move(norm_weights));

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
