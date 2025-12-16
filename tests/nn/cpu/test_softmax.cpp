#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <nn/softmax.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace nn;
using namespace tensor;

TEST(NNSoftmaxTest, Softmax) {
  size_t batch_size = 2;
  size_t vocab_size = 4;

  Tensor<bfloat16, CPU> raw_inputs = arange<bfloat16, CPU>(
      bfloat16(0.0), bfloat16(batch_size * vocab_size), bfloat16(1.0)); // NOLINT
  auto inputs = raw_inputs.view().view_as({batch_size, vocab_size}).copy();

  Softmax softmax;

  auto output = softmax(inputs.view(), 1);

  std::vector<bfloat16> exp{0.0320, 0.0869, 0.2373, 0.6445, 0.0320, 0.0869, 0.2373, 0.6445};

  tensor_is_close<bfloat16>(std::span(exp), output.span(), 1e-2);
}
