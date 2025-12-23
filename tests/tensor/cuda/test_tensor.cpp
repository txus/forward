#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/tensor.hpp>

using namespace tensor;

TEST(TensorCUDATest, FillAndGet) {
  tensor::Tensor<int, tensor::CUDA> gpu_tensor({2, 4});

  gpu_tensor.fill_(4);

  // Copy back to CPU for verification
  auto cpu_tensor = gpu_tensor.cpu();

  std::vector<int> fill_expected = {4, 4, 4, 4, 4, 4, 4, 4};

  tensor_is_close<int>(cpu_tensor.span(), std::span(fill_expected));

  // const auto sliced = tensor.view().get(0);

  // std::vector<int> expected = {4, 4, 4, 4};

  // tensor::Shape expected_shape = {4};

  // EXPECT_EQ(sliced.shape, expected_shape);
  // tensor_is_close<int>(sliced.span(), std::span(expected));
}
