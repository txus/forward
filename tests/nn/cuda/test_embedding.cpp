#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <nn/embedding.hpp>
#include <tensor/tensor.hpp>

using namespace nn;
using namespace tensor;

TEST(NNCUDAEmbeddingTest, Forward) {
  size_t vocab_size = 4;
  size_t hidden_dim = 2;
  size_t batch_size = 1;
  size_t seq_len = 6;

  Tensor<bfloat16, CPU> weights_cpu{{vocab_size, hidden_dim}};
  weights_cpu.fill_(0.0);
  for (int i = 0; i < weights_cpu.size(); ++i) {
    weights_cpu.set_(i, float(i));
  }

  auto weights_ = weights_cpu.cuda();

  Embedding<bfloat16, CUDA> sut;

  sut.load_weights(weights_);

  Tensor<int, CPU> inputs_cpu{{batch_size, seq_len}};
  inputs_cpu.fill_(1);
  for (int i = 0; i < inputs_cpu.size(); ++i) {
    inputs_cpu.set_(i, i % 4);
  }

  auto inputs_ = inputs_cpu.cuda();
  auto inputs = inputs_.view();

  auto result_ = sut.forward(inputs);
  auto result_cpu = result_.cpu();
  auto result = result_cpu.view();

  Shape expected_shape = {batch_size, seq_len, hidden_dim};

  EXPECT_EQ(result.shape, expected_shape);

  // the first 4 tokens tokens in the sequence are literally the first 4 tokens
  // in the vocab
  tensor_is_close<bfloat16>(result.get(0, 0).span(), weights_cpu.view().get(0).span());
  tensor_is_close<bfloat16>(result.get(0, 1).span(), weights_cpu.view().get(1).span());
  tensor_is_close<bfloat16>(result.get(0, 2).span(), weights_cpu.view().get(2).span());
  tensor_is_close<bfloat16>(result.get(0, 3).span(), weights_cpu.view().get(3).span());

  // and now they cycle back -- tokens 5 and 6 in the sequence are 0 and 1 in
  // the vocab
  tensor_is_close<bfloat16>(result.get(0, 4).span(), weights_cpu.view().get(0).span());
  tensor_is_close<bfloat16>(result.get(0, 5).span(), weights_cpu.view().get(1).span());
}
