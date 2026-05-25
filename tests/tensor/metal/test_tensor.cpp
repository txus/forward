#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/tensor.hpp>

using namespace tensor;

TEST(TensorMETALTest, FillAndGet) {
  Tensor<bfloat16, METAL> gpu_tensor({2, 4});

  gpu_tensor.fill_(4);

  // Copy back to CPU for verification
  auto cpu_tensor = gpu_tensor.cpu();

  std::vector<bfloat16> fill_expected = {4, 4, 4, 4, 4, 4, 4, 4};

  tensor_is_close<bfloat16>(cpu_tensor.span(), std::span(fill_expected));
}
