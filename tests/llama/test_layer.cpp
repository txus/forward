#include <common/test_config.h>
#include <common/test_utils.hpp>
#include <gtest/gtest.h>

#include <llama/layer.hpp>

TEST(LayerTest, Forward) {
  llama::Layer layer{};

  const size_t batch_size = 1;
  const size_t seq_len = 2;
  const size_t hidden_dim = 4;

  llama::ModelConfig conf{
      .vocab_size = 128,
      .hidden_dim = hidden_dim,
      .num_hidden_layers = 1,
  };

  auto weights = empty_weights(conf);

  layer.load_weights(weights, 0);

  tensor::Tensor<float> input_{{batch_size, seq_len, hidden_dim}};
  input_.fill_(0.1);
  auto input = input_.view();

  auto output = layer.forward(input);

  fmt::println("Output: {}", output.view());
}
