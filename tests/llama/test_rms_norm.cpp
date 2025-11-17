#include <gtest/gtest.h>
#include <tensor/tensor.hpp>

#include <common/test_utils.hpp>

#include <llama/rms_norm.hpp>

TEST(RMSNormTest, Forward) {
  size_t hidden_dim = 2;
  size_t batch_size = 1;
  size_t seq_len = 4;

  tensor::Tensor<float> weights_{{hidden_dim}};
  weights_.fill_(0.1);

  auto weights = weights_.view();

  tensor::Tensor<float> inputs_{{batch_size, seq_len, hidden_dim}};
  inputs_.fill_(0.0);
  for (int i = 1; i < inputs_.size(); ++i)
    inputs_.set_(i, i);

  auto inputs = inputs_.view();

  llama::RMSNorm sut;
  sut.set_weights(weights);

  auto output = sut.forward(inputs);

  std::vector<float> expected{0,          0.14142136, 0.078446455, 0.11766968,
                              0.08834522, 0.11043153, 0.0920358,   0.1073751};

  tensor_is_close<float>(std::span(expected), output.span());
}
