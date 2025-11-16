#include <gtest/gtest.h>

#include <llama/model.hpp>

TEST(ModelTest, Forward) {
  llama::Model mod{llama::ModelConfig{
      .vocab_size = 128, .hidden_dim = 16, .num_hidden_layers = 1}};

  auto input_ = tensor::Tensor<int>{{1, 4}};

  auto v = input_.view();

  auto output = mod.forward(v);
}
