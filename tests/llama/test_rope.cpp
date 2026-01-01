#include <common/test_config.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <llama/layer.hpp>
#include <llama/rope.hpp>

using namespace llama;
using namespace tensor;

const size_t head_dim = 4;
const size_t seq_len = 6;
const float theta = 10000.0;

TEST(LlamaRoPETest, PrecomputeRopeValues) {
  llama::ModelConfig conf{
      .head_dim = head_dim, .rope_theta = theta, .max_position_embeddings = seq_len};

  RoPE<bfloat16, CPU> rope{conf};

  auto cos = rope.cos();
  auto sin = rope.sin();

  std::vector<float> expected_cos{1.0000,  1.0000, 1.0000,  1.0000, 0.5403,  0.9999,
                                  0.5403,  0.9999, -0.4161, 0.9998, -0.4161, 0.9998,
                                  -0.9900, 0.9996, -0.9900, 0.9996, -0.6536, 0.9992,
                                  -0.6536, 0.9992, 0.2837,  0.9988, 0.2837,  0.9988};

  tensor_is_close<float>(cos.span(), std::span(expected_cos));

  std::vector<float> expected_sin{0.0000,  0.0000, 0.0000,  0.0000, 0.8415,  0.0100,
                                  0.8415,  0.0100, 0.9093,  0.0200, 0.9093,  0.0200,
                                  0.1411,  0.0300, 0.1411,  0.0300, -0.7568, 0.0400,
                                  -0.7568, 0.0400, -0.9589, 0.0500, -0.9589, 0.0500};

  tensor_is_close<float>(sin.span(), std::span(expected_sin));
}

TEST(LlamaRoPETest, Forward) {
  llama::ModelConfig conf{
      .head_dim = head_dim, .rope_theta = theta, .max_position_embeddings = seq_len};

  RoPE<bfloat16, CPU> rope{conf};

  Tensor<bfloat16, CPU> inputs{{1, 2, seq_len, head_dim}};
  inputs.fill_(1.0);

  Tensor<bfloat16, CPU> outputs = rope.forward(inputs.view());

  std::vector<bfloat16> expected{
      1.0000, 1.0000, 1.0000,  1.0000, -0.3008, 0.9883, 1.3828,  1.0078, -1.3281, 0.9805,
      0.4922, 1.0234, -1.1328, 0.9688, -0.8477, 1.0312, 0.1030,  0.9609, -1.4141, 1.0391, // NOLINT
      1.2422, 0.9492, -0.6758, 1.0469, 1.0000,  1.0000, 1.0000,  1.0000, -0.3008, 0.9883,
      1.3828, 1.0078, -1.3281, 0.9805, 0.4922,  1.0234, -1.1328, 0.9688, -0.8477, 1.0312,
      0.1030, 0.9609, -1.4141, 1.0391, 1.2422,  0.9492, -0.6758, 1.0469}; // NOLINT

  tensor_is_close<bfloat16>(outputs.view().span(), std::span(expected), 1e-2);
}

#ifdef BACKEND_CUDA
TEST(LlamaCUDARoPETest, PrecomputeRopeValues) {
  SKIP_IF_NO_GPU();
  llama::ModelConfig conf{
      .head_dim = head_dim, .rope_theta = theta, .max_position_embeddings = seq_len};

  RoPE<bfloat16, CUDA> rope{conf};

  auto cos = rope.cos();
  auto sin = rope.sin();

  auto cos_cpu = copy(cos).cpu();
  auto sin_cpu = copy(sin).cpu();

  std::vector<float> expected_cos{1.0000,  1.0000, 1.0000,  1.0000, 0.5403,  0.9999,
                                  0.5403,  0.9999, -0.4161, 0.9998, -0.4161, 0.9998,
                                  -0.9900, 0.9996, -0.9900, 0.9996, -0.6536, 0.9992,
                                  -0.6536, 0.9992, 0.2837,  0.9988, 0.2837,  0.9988};

  tensor_is_close<float>(cos_cpu.span(), std::span(expected_cos));

  std::vector<float> expected_sin{0.0000,  0.0000, 0.0000,  0.0000, 0.8415,  0.0100,
                                  0.8415,  0.0100, 0.9093,  0.0200, 0.9093,  0.0200,
                                  0.1411,  0.0300, 0.1411,  0.0300, -0.7568, 0.0400,
                                  -0.7568, 0.0400, -0.9589, 0.0500, -0.9589, 0.0500};

  tensor_is_close<float>(sin_cpu.span(), std::span(expected_sin));
}

TEST(LlamaCUDARoPETest, Forward) {
  SKIP_IF_NO_GPU();
  llama::ModelConfig conf{
      .head_dim = head_dim, .rope_theta = theta, .max_position_embeddings = seq_len};

  RoPE<bfloat16, CUDA> rope{conf};

  Tensor<bfloat16, CUDA> inputs{{1, 2, seq_len, head_dim}};
  inputs.fill_(1.0);

  Tensor<bfloat16, CUDA> outputs = rope.forward(inputs.view());

  auto outputs_cpu = outputs.cpu();

  std::vector<bfloat16> expected{
      1.0000, 1.0000, 1.0000,  1.0000, -0.3008, 0.9883, 1.3828,  1.0078, -1.3281, 0.9805,
      0.4922, 1.0234, -1.1328, 0.9688, -0.8477, 1.0312, 0.1030,  0.9609, -1.4141, 1.0391, // NOLINT
      1.2422, 0.9492, -0.6758, 1.0469, 1.0000,  1.0000, 1.0000,  1.0000, -0.3008, 0.9883,
      1.3828, 1.0078, -1.3281, 0.9805, 0.4922,  1.0234, -1.1328, 0.9688, -0.8477, 1.0312,
      0.1030, 0.9609, -1.4141, 1.0391, 1.2422,  0.9492, -0.6758, 1.0469}; // NOLINT

  tensor_is_close<bfloat16>(outputs_cpu.view().span(), std::span(expected), 1e-2);
}
#endif
