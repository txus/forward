#include <common/test_config.h>
#include <common/test_utils.hpp>
#include <gtest/gtest.h>

#include <llama/model.hpp>

TEST(ModelTest, Forward) {
  llama::Model mod{llama::ModelConfig{
      .vocab_size = 128,
      .hidden_dim = 32,
      .num_hidden_layers = 1,
  }};

  auto weights = empty_weights(mod.config);

  mod.load_weights(weights);

  auto input_ = tensor::Tensor<int>{{1, 4}};

  auto v = input_.view();

  auto output = mod.forward(v);
}
