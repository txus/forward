#include <gtest/gtest.h>
#include <tensor/tensor.hpp>

#include <common/test_utils.hpp>

#include <llama/embedding.hpp>

TEST(EmbeddingTest, Forward) {
  size_t vocab_size = 4;
  size_t hidden_dim = 2;
  size_t batch_size = 1;
  size_t seq_len = 6;

  tensor::Tensor<float> weights_{{vocab_size, hidden_dim}};
  weights_.fill_(0.0);
  for (int i = 0; i < weights_.size(); ++i)
    weights_.set_(i, i);

  auto weights = weights_.view();

  llama::Embedding sut;
  sut.set_weights(weights);

  tensor::Tensor<int> inputs_{{batch_size, seq_len}};
  inputs_.fill_(1);
  for (int i = 0; i < inputs_.size(); ++i)
    inputs_.set_(i, i % 4);

  auto inputs = inputs_.view();

  auto result_ = sut.forward(inputs);
  auto result = result_.view();

  tensor::Shape expected_shape = {batch_size, seq_len, hidden_dim};

  EXPECT_EQ(result.shape, expected_shape);

  // the first 4 tokens tokens in the sequence are literally the first 4 tokens
  // in the vocab
  tensor_is_close<float>(result.get(0, 0).span(), weights.get(0).span());
  tensor_is_close<float>(result.get(0, 1).span(), weights.get(1).span());
  tensor_is_close<float>(result.get(0, 2).span(), weights.get(2).span());
  tensor_is_close<float>(result.get(0, 3).span(), weights.get(3).span());

  // and now they cycle back -- tokens 5 and 6 in the sequence are 0 and 1 in
  // the vocab
  tensor_is_close<float>(result.get(0, 4).span(), weights.get(0).span());
  tensor_is_close<float>(result.get(0, 5).span(), weights.get(1).span());
}
