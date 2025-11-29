#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <nn/rms_norm.hpp>
#include <tensor/tensor.hpp>

using namespace nn;
using namespace tensor;

TEST(NNRMSNormTest, Forward) {
  size_t hidden_dim = 2;
  size_t batch_size = 1;
  size_t seq_len = 4;

  Tensor<bfloat16, CPU> weights_{{hidden_dim}};
  weights_.fill_(0.1001);
  auto weights = weights_.view();

  Tensor<bfloat16, CPU> inputs_{{batch_size, seq_len, hidden_dim}};
  // arange
  for (int i = 0; i < inputs_.size(); ++i) {
    inputs_.span()[i] = float(i);
  }
  auto inputs = inputs_.view();

  RMSNorm<bfloat16, CPU> sut;
  sut.set_weights(weights);

  auto output = sut.forward(inputs);

  std::vector<bfloat16> expected_vec{0.0000, 0.1416, 0.0786, 0.1177,
                                     0.0884, 0.1104, 0.0923, 0.1074};

  tensor_is_close<bfloat16>(std::span(expected_vec), output.span());
}
