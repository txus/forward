#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/config.hpp>
#include <llama/grouped_query_attention.hpp>
#include <tensor/loader.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaGQATest, Parity) {
  Loader<bfloat16, CPU> act_loader(TEST_ACTIVATIONS_PATH);
  auto input_activations_ = act_loader.load("layers.0.input_layernorm");
  auto input_activations = copy(input_activations_.view());
  auto output_activations = act_loader.load("layers.0.self_attn.o_proj");

  Loader<bfloat16, CPU> weights_loader(TEST_WEIGHTS_PATH);

  llama::ModelConfig conf = load_config(TEST_CONFIG_PATH);

  GroupedQueryAttention<bfloat16, CPU> gqa{conf};

  gqa.load_weights(weights_loader, 0);

  auto seq_len = input_activations.shape()[1];

  RoPE<bfloat16, CPU> rope{conf};

  auto attn_mask = causal_attention_mask<int, CPU>(seq_len);

  auto output = gqa.forward(input_activations.view(), attn_mask.view(), rope);

  tensor_is_close<bfloat16>(output.view().span(), output_activations.span());
}

TEST(LlamaGQATest, ParityWithKVCache) {
  Loader<bfloat16, CPU> act_loader(TEST_ACTIVATIONS_PATH);
  auto input_activations_ = act_loader.load("layers.0.input_layernorm");
  auto input_activations = copy(input_activations_.view());
  auto output_activations = act_loader.load("layers.0.self_attn.o_proj");

  Loader<bfloat16, CPU> weights_loader(TEST_WEIGHTS_PATH);

  llama::ModelConfig conf = load_config(TEST_CONFIG_PATH);

  GroupedQueryAttention<bfloat16, CPU> gqa{conf, 128};

  gqa.load_weights(weights_loader, 0);

  auto seq_len = input_activations.shape()[1];

  RoPE<bfloat16, CPU> rope{conf};

  auto attn_mask = causal_attention_mask<int, CPU>(seq_len);

  EXPECT_EQ(gqa.get_cache_size(), 0);

  // prefill 3 tokens
  auto first_3_input_activations = slice(input_activations.view(), 1, 0, 3);
  auto prefill_output = gqa.forward(first_3_input_activations.view(), attn_mask.view(), rope);
  auto expected_prefill_output = slice(output_activations.view(), 1, 0, 3);
  tensor_is_close<bfloat16>(prefill_output.view().span(), expected_prefill_output.span());

  EXPECT_EQ(gqa.get_cache_size(), 3);

  // decode a 4th
  auto fourth_activation = slice(input_activations.view(), 1, 3, 4);
  auto decode_output = gqa.forward(fourth_activation.view(), attn_mask.view(), rope);
  auto expected_decode_output = slice(output_activations.view(), 1, 3, 4);

  tensor_is_close<bfloat16>(decode_output.view().span(), expected_decode_output.span());

  EXPECT_EQ(gqa.get_cache_size(), 4);
}
