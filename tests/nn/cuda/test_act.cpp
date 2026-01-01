#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <nn/act.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace nn;
using namespace tensor;

TEST(NNCUDAActTest, SiLU) {
  SKIP_IF_NO_GPU();
  size_t length = 2;

  Tensor<bfloat16, CUDA> inputs_ =
      arange<bfloat16, CUDA>(bfloat16(0.0), bfloat16(length), bfloat16(1.0)); // NOLINT
  auto inputs = inputs_.view();

  SiLU silu;

  auto output_ = silu(inputs);
  auto output = output_.cpu();

  std::vector<bfloat16> expected_vec{0.0000, 0.7305};

  tensor_is_close<bfloat16>(std::span(expected_vec), output.span());
}
