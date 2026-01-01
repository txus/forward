#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/layer.hpp>
#include <llama/model.hpp>
#include <tensor/ops.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaLayerTest, Parity) {
  Loader<bfloat16, CPU> act_loader(TEST_ACTIVATIONS_PATH);
  auto input_activations_ = act_loader.load("embed_tokens");
  auto input_activations = copy(input_activations_.view());
  auto output_activations = act_loader.load("layers.0");

  Loader<bfloat16, CPU> weights_loader(TEST_WEIGHTS_PATH);

  llama::ModelConfig conf = load_config(TEST_CONFIG_PATH);

  fmt::println("INPUT SHAPE {}", input_activations.shape());
  fmt::println("SEQ LEN {}", input_activations.shape()[1]);

  auto seq_len = input_activations.shape()[1];

  Layer<bfloat16, CPU> layer{
      conf,
  };

  layer.load_weights(weights_loader, 0);

  RoPE<bfloat16, CPU> rope{conf};

  auto attn_mask = causal_attention_mask<int, CPU>(seq_len);

  auto output = layer.forward(input_activations.view(), attn_mask.view(), rope);

  tensor_is_close<bfloat16>(output.view().span(), output_activations.span(), 1e-02);
}
