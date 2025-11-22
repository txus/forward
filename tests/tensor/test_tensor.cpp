#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/tensor.hpp>

TEST(TensorTest, Stride) {
  tensor::Shape shape{2, 4};

  EXPECT_EQ(tensor::stride(shape, 0), 4);
  EXPECT_EQ(tensor::stride(shape, 1), 1);
}

TEST(TensorTest, FillAndGet) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  tensor.fill_(4);

  std::vector<int> fill_expected = {4, 4, 4, 4, 4, 4, 4, 4};

  tensor_is_close<int>(tensor.span(), std::span(fill_expected));

  const auto sliced = tensor.view().get(0);

  std::vector<int> expected = {4, 4, 4, 4};

  tensor::Shape expected_shape = {4};

  EXPECT_EQ(sliced.shape, expected_shape);
  tensor_is_close<int>(sliced.span(), std::span(expected));
}

TEST(TensorTest, Slice3d) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4, 3});
  const auto first = tensor.view().get(0);
  tensor::Shape shape = {4, 3};

  EXPECT_EQ(first.shape, shape);
}

TEST(TensorTest, Set) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  tensor.fill_(0);

  tensor.set_(2, 6);

  std::vector<int> expected = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor_is_close<int>(tensor.span(), std::span(expected));
}

TEST(TensorTest, ConstructWithData) {
  std::vector<int> data = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor::Tensor<int, tensor::CPU> tensor({2, 4}, std::move(data));

  std::vector<int> expected = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor_is_close<int>(tensor.span(), std::span(expected));
}

TEST(TensorTest, Copy) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  auto view = tensor.view();

  auto new_t = view.copy();
  new_t.set_(0, 6);
  tensor.set_(0, 7);

  EXPECT_EQ(tensor.at(0), 7);
  EXPECT_EQ(new_t.at(0), 6);
}

TEST(TensorTest, CopySlice) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  auto view = tensor.view().get(0);

  auto new_t = view.copy();
  new_t.set_(0, 6);
  tensor.set_(0, 7);

  EXPECT_EQ(tensor.at(0), 7);
  EXPECT_EQ(new_t.at(0), 6);
}
