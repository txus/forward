#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/config.hpp>
#include <llama/mlp.hpp>
#include <tensor/loader.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaMLPTest, Parity) {
  Loader<bfloat16, CPU> act_loader(TEST_ACTIVATIONS_PATH);
  auto input_activations_ = act_loader.load("layers.0.post_attention_layernorm");
  auto input_activations = copy(input_activations_.view());
  auto output_activations = act_loader.load("layers.0.mlp.down_proj");

  fmt::println("TEST INPUT (after layernorm) {}", input_activations.view());

  Loader<bfloat16, CPU> weights_loader(TEST_WEIGHTS_PATH);

  llama::ModelConfig conf = load_config(TEST_CONFIG_PATH);

  MLP<bfloat16, CPU> mlp{conf};

  mlp.load_weights(weights_loader, 0);

  auto output = mlp.forward(input_activations.view());

  tensor_is_close<bfloat16>(output.view().span(), output_activations.span());
}
