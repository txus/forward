#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/tensor.hpp>

using namespace nn;
using namespace tensor;

TEST(NNCUDARMSNormTest, Forward) {
  SKIP_IF_NO_GPU();
  size_t hidden_dim = 2;
  size_t batch_size = 1;
  size_t seq_len = 4;

  Tensor<bfloat16, CUDA> weights{{hidden_dim}};
  weights.fill_(0.1001);

  Tensor<bfloat16, CPU> inputs_cpu{{batch_size, seq_len, hidden_dim}};
  // arange
  for (int i = 0; i < inputs_cpu.size(); ++i) {
    inputs_cpu.span()[i] = float(i);
  }
  auto inputs_ = inputs_cpu.cuda();
  auto inputs = inputs_.view();

  RMSNorm<bfloat16, CUDA> sut(1e-5);
  sut.load_weights(weights);

  auto output = sut.forward(inputs);

  auto output_cpu = output.cpu();

  std::vector<bfloat16> expected_vec{0.0000, 0.1416, 0.0786, 0.1177,
                                     0.0884, 0.1104, 0.0923, 0.1074};

  tensor_is_close<bfloat16>(std::span(expected_vec), output_cpu.span());
}
