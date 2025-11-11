#include "forward/tensor.hpp"
#include <gtest/gtest.h>

#include <forward/nn/embedding.hpp>

TEST(EmbeddingTest, Forward) {
  size_t vocab_size = 4;
  size_t hidden_dim = 2;
  size_t batch_size = 1;
  size_t seq_len = 6;

  tensor::Tensor<float> weights{{vocab_size, hidden_dim}};
  weights.fill_(0.0);
  for (int i = 0; i < weights.size(); ++i)
    weights.set_(i, i);

  nn::Embedding sut{weights};

  tensor::Tensor<int> inputs{{batch_size, seq_len}};
  inputs.fill_(1);
  for (int i = 0; i < inputs.size(); ++i)
    inputs.set_(i, i % 4);

  auto result = sut.forward(inputs);

  tensor::Shape expected_shape = {batch_size, seq_len, hidden_dim};

  EXPECT_EQ(result.shape(), expected_shape);

  auto tolerance = 0.001;

  auto first_batch_elem = result.slice(0);

  auto weights_tok0 = weights.slice(0);
  auto weights_tok1 = weights.slice(1);
  auto weights_tok2 = weights.slice(2);
  auto weights_tok3 = weights.slice(3);

  auto first_seq_elem = first_batch_elem.slice(0);
  auto second_seq_elem = first_batch_elem.slice(1);
  auto third_seq_elem = first_batch_elem.slice(2);
  auto fourth_seq_elem = first_batch_elem.slice(3);
  auto fifth_seq_elem = first_batch_elem.slice(4);
  auto sixth_seq_elem = first_batch_elem.slice(5);

  tensor::assert_all_close(&weights_tok0, &first_seq_elem, tolerance,
                           "1st elem wasn't embedded properly");
  tensor::assert_all_close(&weights_tok1, &second_seq_elem, tolerance,
                           "2nd elem wasn't embedded properly");
  tensor::assert_all_close(&weights_tok2, &third_seq_elem, tolerance,
                           "3rd elem wasn't embedded properly");
  tensor::assert_all_close(&weights_tok3, &fourth_seq_elem, tolerance,
                           "4th elem wasn't embedded properly");
  // wrap around
  tensor::assert_all_close(&weights_tok0, &fifth_seq_elem, tolerance,
                           "5th elem wasn't embedded properly");
  tensor::assert_all_close(&weights_tok1, &sixth_seq_elem, tolerance,
                           "6th elem wasn't embedded properly");
}
