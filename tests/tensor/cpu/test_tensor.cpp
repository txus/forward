#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/tensor.hpp>

TEST(TensorCPUTest, FillAndGet) {
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

TEST(TensorCPUTest, Slice3d) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4, 3});
  const auto first = tensor.view().get(0);
  tensor::Shape shape = {4, 3};

  EXPECT_EQ(first.shape, shape);
}

TEST(TensorCPUTest, Set) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  tensor.fill_(0);

  tensor.set_(2, 6);

  std::vector<int> expected = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor_is_close<int>(tensor.span(), std::span(expected));
}

TEST(TensorCPUTest, ConstructWithData) {
  std::vector<int> data = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor::Tensor<int, tensor::CPU> tensor({2, 4}, std::move(data));

  std::vector<int> expected = {0, 0, 6, 0, 0, 0, 0, 0};

  tensor_is_close<int>(tensor.span(), std::span(expected));
}

TEST(TensorCPUTest, Copy) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  auto view = tensor.view();

  auto new_t = view.copy();
  new_t.set_(0, 6);
  tensor.set_(0, 7);

  EXPECT_EQ(tensor.at(0), 7);
  EXPECT_EQ(new_t.at(0), 6);
}

TEST(TensorCPUTest, CopySlice) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  auto view = tensor.view().get(0);

  auto new_t = view.copy();
  new_t.set_(0, 6);
  tensor.set_(0, 7);

  EXPECT_EQ(tensor.at(0), 7);
  EXPECT_EQ(new_t.at(0), 6);
}

TEST(TensorCPUTest, Transpose_Contiguous) {
  // Create a 2x3x4 tensor with sequential values
  tensor::Tensor<int, tensor::CPU> tensor({2, 3, 4});
  for (int i = 0; i < 24; ++i) {
    tensor.set_(i, i);
  }

  // Transpose dimensions 1 and 2: (2, 3, 4) -> (2, 4, 3)
  auto view = tensor.view();
  view.transpose(1, 2);

  // Verify shape changed
  tensor::Shape expected_shape = {2, 4, 3};
  EXPECT_EQ(view.shape, expected_shape);

  // Make it contiguous
  auto contiguous_tensor = view.contiguous();

  // Verify the data is properly materialized
  // Original layout: [0,1,2,3, 4,5,6,7, 8,9,10,11 | 12,13,14,15, 16,17,18,19, 20,21,22,23]
  // After transpose(1,2):
  //   Position [0,0,:] should be [0, 4, 8] (first element of each of the 3 inner arrays)
  //   Position [0,1,:] should be [1, 5, 9] (second element of each of the 3 inner arrays)
  EXPECT_EQ(contiguous_tensor.at(0), 0); // [0,0,0]
  EXPECT_EQ(contiguous_tensor.at(1), 4); // [0,0,1]
  EXPECT_EQ(contiguous_tensor.at(2), 8); // [0,0,2]
  EXPECT_EQ(contiguous_tensor.at(3), 1); // [0,1,0]
  EXPECT_EQ(contiguous_tensor.at(4), 5); // [0,1,1]
  EXPECT_EQ(contiguous_tensor.at(5), 9); // [0,1,2]
}

TEST(TensorCPUTest, Reshape_NonContiguous) {
  // Create a 2x3x4 tensor
  tensor::Tensor<int, tensor::CPU> tensor({2, 3, 4});
  for (int i = 0; i < 24; ++i) {
    tensor.set_(i, i);
  }

  // Transpose to make it non-contiguous
  auto view = tensor.view();
  view.transpose(1, 2); // (2, 3, 4) -> (2, 4, 3)

  // reshape should handle the non-contiguous data
  auto reshaped = view.reshape({2, 12});

  // After proper handling, the first row should be [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
  EXPECT_EQ(reshaped.at(0), 0);
  EXPECT_EQ(reshaped.at(1), 4);
  EXPECT_EQ(reshaped.at(2), 8);
  EXPECT_EQ(reshaped.at(3), 1);
  EXPECT_EQ(reshaped.at(4), 5);
  EXPECT_EQ(reshaped.at(5), 9);
}

TEST(TensorCPUTest, Reshape_TransposeLike_GQA) {
  // Simulate the exact GQA scenario: (1, 32, 6, 64) -> transpose(1,2) -> (1, 6, 32, 64) ->
  // reshape(1, 6, 2048) Simplified to (1, 2, 3, 4) -> transpose(1,2) -> (1, 3, 2, 4) -> reshape(1,
  // 3, 8)
  tensor::Tensor<int, tensor::CPU> tensor({1, 2, 3, 4});
  for (int i = 0; i < 24; ++i) {
    tensor.set_(i, i);
  }

  auto view = tensor.view();
  view.transpose(1, 2); // (1, 2, 3, 4) -> (1, 3, 2, 4)

  fmt::print("Before reshape: is_contiguous={}\n", view.is_contiguous());

  auto reshaped = view.reshape({1, 3, 8});

  fmt::print("Reshaped tensor values [0-7]: [");
  for (int i = 0; i < 8; ++i) {
    fmt::print("{}", reshaped.at(i));
    if (i < 7)
      fmt::print(", ");
  }
  fmt::println("]");

  // Check that indices 0-3 (first head, position 0) differ from indices 4-7 (second head, position
  // 0) Original layout: [0,1,2,3, 4,5,6,7, 8,9,10,11 | 12,13,14,15, 16,17,18,19, 20,21,22,23] After
  // transpose(1,2), position [0,0,:] should be [0,1,2,3, 12,13,14,15] (first elements of each 2
  // heads)
  EXPECT_EQ(reshaped.at(0), 0); // [0,0,0]
  EXPECT_EQ(reshaped.at(1), 1); // [0,0,1]
  EXPECT_EQ(reshaped.at(2), 2); // [0,0,2]
  EXPECT_EQ(reshaped.at(3), 3); // [0,0,3]
  EXPECT_EQ(reshaped.at(4),
            12); // [0,0,4] - THIS IS THE KEY TEST: should be from head 1, not head 0
  EXPECT_EQ(reshaped.at(5), 13); // [0,0,5]
  EXPECT_EQ(reshaped.at(6), 14); // [0,0,6]
  EXPECT_EQ(reshaped.at(7), 15); // [0,0,7]
}
