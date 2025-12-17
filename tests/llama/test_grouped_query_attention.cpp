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
  auto input_activations = act_loader.load("layers.0.input_layernorm").copy();
  auto output_activations = act_loader.load("layers.0.self_attn.o_proj");

  Loader<bfloat16, CPU> weights_loader(TEST_MODEL_PATH "/model.safetensors");

  llama::ModelConfig conf = load_config(std::string(TEST_MODEL_PATH "/config.json"));

  GroupedQueryAttention<bfloat16, CPU> gqa{conf};

  gqa.load_weights(weights_loader, 0);

  auto seq_len = input_activations.shape()[1];

  RoPE<bfloat16, CPU> rope{conf};

  auto attn_mask = causal_attention_mask<int, CPU>(seq_len);

  auto output = gqa.forward(input_activations.view(), attn_mask.view(), rope);

  tensor_is_close<bfloat16>(output.view().span(), output_activations.span());
}
