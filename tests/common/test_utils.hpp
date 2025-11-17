#pragma once

#include <cmath>
#include <gtest/gtest.h>
#include <llama/model.hpp>
#include <span>
#include <tensor/tensor.hpp>

inline std::unordered_map<std::string, tensor::Tensor<float>>
empty_weights(const llama::ModelConfig &config) {
  std::unordered_map<std::string, tensor::Tensor<float>> out;

  {
    std::vector<size_t> shape{{config.vocab_size, config.hidden_dim}};

    tensor::Tensor<float> weights{shape};
    weights.fill_(0.5);

    out.insert_or_assign("model.embed_tokens.weight", std::move(weights));
  }

  for (int l = 0; l < config.num_hidden_layers; ++l) {
    std::vector<size_t> shape{config.hidden_dim};

    const std::string key =
        fmt::format("model.layers.{}.input_layernorm.weight", l);

    tensor::Tensor<float> weights{shape};
    weights.fill_(0.1);

    out.insert_or_assign(key, std::move(weights));
  }

  return out;
}

template <typename T>
void tensor_is_close(std::span<const T> a, std::span<const T> b,
                     float atol = float(1e-5), float rtol = float(1e-5)) {
  ASSERT_EQ(a.size(), b.size()) << "Span sizes differ";

  for (size_t i = 0; i < a.size(); ++i) {
    T diff = std::fabs(a[i] - b[i]);
    T limit = atol + rtol * std::fabs(b[i]);
    ASSERT_LE(diff, limit) << "Mismatch at index " << i << ": a=" << a[i]
                           << " b=" << b[i] << " diff=" << diff
                           << " limit=" << limit;
  }
}
