#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/layer.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaLayerTest, Forward) {
  const size_t batch_size = 1;
  const size_t seq_len = 2;
  const size_t hidden_size = 4;
  const size_t intermediate_size = 8;

  llama::ModelConfig conf{
      .vocab_size = 128,
      .hidden_size = hidden_size,
      .intermediate_size = intermediate_size,
      .num_hidden_layers = 1,
      .hidden_act = "silu",
  };

  Layer<bfloat16, CPU> layer{conf};

  auto weights = empty_weights(conf);

  layer.load_weights(weights, 0);

  Tensor<bfloat16, CPU> input_{{batch_size, seq_len, hidden_size}};
  input_.fill_(0.1);
  auto input = input_.view();

  auto output = layer.forward(input);

  fmt::println("Output: {}", output.view());
}
