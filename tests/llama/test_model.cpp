#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/model.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaModelTest, Forward) {
  Model<bfloat16, CPU> mod{ModelConfig{
      .vocab_size = 128,
      .hidden_dim = 32,
      .num_hidden_layers = 1,
  }};

  auto weights = empty_weights(mod.config);

  mod.load_weights(weights);

  auto input_ = Tensor<int, CPU>{{1, 4}};

  auto view = input_.view();

  auto output = mod.forward(view);
}
