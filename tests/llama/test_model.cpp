#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/model.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaModelTest, Forward) {
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

  Loader<bfloat16, CPU> weights_loader(TEST_MODEL_PATH "/model.safetensors");

  Model<bfloat16, CPU> mod{conf};

  mod.load_weights(weights_loader);

  auto input_ = Tensor<int, CPU>{{1, 4}};

  auto view = input_.view();

  auto output = mod.forward(view);
}
