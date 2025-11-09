#include <gtest/gtest.h>

#include <forward/tensor.hpp>

TEST(TensorTest, Stride) {
  tensor::Tensor<int> t({2, 4});

  EXPECT_EQ(t.stride(0), 4);
  EXPECT_EQ(t.stride(1), 1);
}

TEST(TensorTest, FillAndSlice) {
  tensor::Tensor<int> t({2, 4});

  t.fill_(4);

  std::vector<int> fill_expected = {4, 4, 4, 4, 4, 4, 4, 4};
  EXPECT_EQ(t.raw(), fill_expected);

  std::vector<int> expected = {4, 4, 4, 4};

  auto actual = t.slice(0).raw();

  EXPECT_EQ(actual, expected);
}
