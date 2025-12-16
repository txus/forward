#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <nn/act.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace nn;
using namespace tensor;

TEST(NNActTest, SiLU) {
  size_t length = 2;

  Tensor<bfloat16, CPU> inputs_ =
      arange<bfloat16, CPU>(bfloat16(0.0), bfloat16(length), bfloat16(1.0)); // NOLINT
  auto inputs = inputs_.view();

  SiLU silu;

  auto output = silu(inputs);

  fmt::println("Inputs {}", inputs);

  std::vector<bfloat16> expected_vec{0.0000, 0.7305};

  tensor_is_close<bfloat16>(std::span(expected_vec), output.span());
}
