#include <common/test_config.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/layer.hpp>
#include <llama/rope.hpp>

using namespace llama;
using namespace tensor;

TEST(LlamaRoPETest, PrecomputeRopeValues) {
  llama::ModelConfig conf{.rope_theta = 10000.0, .max_position_embeddings = 4096};

  RoPE<bfloat16, CPU> rope{conf};

  auto cos = rope.cos();

  fmt::println("Output: {}", cos);
}
