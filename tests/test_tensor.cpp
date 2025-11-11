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

  auto actual = t.slice(0);

  tensor::Shape expected_shape = {4};
  EXPECT_EQ(actual.shape(), expected_shape);
  EXPECT_EQ(actual.raw(), expected);
}

TEST(TensorTest, Slice3d) {
  tensor::Tensor<int> t({2, 4, 3});
  auto first = t.slice(0);
  tensor::Shape shape = {4, 3};

  EXPECT_EQ(first.shape(), shape);
}

TEST(TensorTest, Set) {
  tensor::Tensor<int> t({2, 4});

  t.fill_(0);

  t.set_(2, 6);

  std::vector<int> expected = {0, 0, 6, 0, 0, 0, 0, 0};
  EXPECT_EQ(t.raw(), expected);
}

TEST(TensorTest, ConstructWithData) {
  std::vector<int> data = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor::Tensor<int> t({2, 4}, std::move(data));

  std::vector<int> expected = {0, 0, 6, 0, 0, 0, 0, 0};

  EXPECT_EQ(t.raw(), expected);
}
