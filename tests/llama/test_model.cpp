#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/model.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaModelTest, Forward) {
  const size_t batch_size = 1;
  const size_t seq_len = 4;

  Loader<bfloat16, CPU> weights_loader(TEST_WEIGHTS_PATH);
  llama::ModelConfig conf = load_config(TEST_CONFIG_PATH);

  Model<bfloat16, CPU> mod{
      conf,
      128,
  };

  mod.load_weights(weights_loader);

  auto input_ = Tensor<int, CPU>{{batch_size, seq_len}};

  auto view = input_.view();

  auto output = mod.forward(view);
}
