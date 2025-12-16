#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/layer.hpp>
#include <llama/model.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaLayerTest, Forward) {
  const size_t batch_size = 1;
  const size_t seq_len = 4;
  const size_t hidden_size = 16;

  const size_t head_dim = 4;
  const size_t num_attention_heads = 4;
  const size_t num_kv_heads = 1;

  llama::ModelConfig conf{.vocab_size = 128,
                          .head_dim = head_dim,
                          .rope_theta = 10000,
                          .hidden_size = hidden_size,
                          .max_position_embeddings = 128,
                          .num_attention_heads = num_attention_heads,
                          .num_hidden_layers = 1,
                          .num_key_value_heads = num_kv_heads};

  Layer<bfloat16, CPU> layer{conf};

  auto weights = empty_weights(conf);

  layer.load_weights(weights, 0);

  Tensor<bfloat16, CPU> input_{{batch_size, seq_len, hidden_size}};
  input_.fill_(0.1);
  auto input = input_.view();

  RoPE<bfloat16, CPU> rope{conf};

  auto attn_mask = causal_attention_mask<int, CPU>(seq_len);

  auto output = layer.forward(input, attn_mask.view(), rope);

  fmt::println("Output: {}", output.view());
}
