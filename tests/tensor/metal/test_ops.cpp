#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/ops.hpp>

using namespace tensor;

TEST(TensorMETALTest, Arange) {
  Tensor<int, METAL> result = arange<int, METAL>(0, 10, 2);

  auto cpu = result.cpu();

  std::vector<int> exp = {0, 2, 4, 6, 8};

  tensor_is_close<int>(cpu.span(), std::span(exp));
}
