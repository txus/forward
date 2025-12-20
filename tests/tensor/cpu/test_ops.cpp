#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/ops.hpp>

using namespace tensor;

TEST(TensorCPUTest, Arange) {
  Tensor<int, CPU> result = arange<int, CPU>(0, 10, 2);

  std::vector<int> exp = {0, 2, 4, 6, 8};

  tensor_is_close<int>(result.span(), std::span(exp));
}

TEST(TensorCPUTest, AddInt) {
  Tensor<int, CPU> tensor_a({2, 4});
  Tensor<int, CPU> tensor_b({2, 4});
  Tensor<int, CPU> exp({2, 4});

  tensor_a.fill_(4);
  tensor_b.fill_(3);
  exp.fill_(7);

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<int, CPU> result = add(a_v, b_v);

  tensor_is_close<int>(result.span(), exp.span());
}

TEST(TensorCPUTest, AddBF16) {
  Tensor<bfloat16, CPU> tensor_a({2, 4});
  Tensor<bfloat16, CPU> tensor_b({2, 4});
  Tensor<bfloat16, CPU> exp({2, 4});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));
  exp.fill_(bfloat16(7.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<bfloat16, CPU> result = add(a_v, b_v);

  tensor_is_close<bfloat16>(result.span(), exp.span());
}

TEST(TensorCPUTest, PowBF16) {
  Tensor<bfloat16, CPU> tensor({2, 2});
  tensor.fill_(2.0);

  Tensor<bfloat16, CPU> result = pow<bfloat16>(3.0, tensor.view());

  std::vector<bfloat16> exp = {9, 9, 9, 9};

  tensor_is_close<bfloat16>(result.span(), std::span(exp));
}

TEST(TensorCPUTest, MatmulBF16) {
  Tensor<bfloat16, CPU> tensor_a({2, 4});
  Tensor<bfloat16, CPU> tensor_b({4, 2});
  Tensor<bfloat16, CPU> exp({2, 2});

  tensor_a.fill_(bfloat16(2.0));
  tensor_b.fill_(bfloat16(3.0));
  exp.fill_(bfloat16(24.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<bfloat16, CPU> result = matmul(a_v, b_v);

  tensor_is_close<bfloat16>(result.span(), exp.span());
}

TEST(TensorCPUTest, CatF16) {
  Tensor<bfloat16, CPU> tensor_a({2, 4});
  Tensor<bfloat16, CPU> tensor_b({2, 2});

  tensor_a.fill_(bfloat16(2.0));
  tensor_b.fill_(bfloat16(3.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  std::vector<bfloat16> exp = {2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0};

  Tensor<bfloat16, CPU> result = cat(a_v, b_v, 1);

  Shape expected_shape = {2, 6};

  EXPECT_EQ(result.shape(), expected_shape);

  tensor_is_close<bfloat16>(result.span(), std::span(exp));
}

TEST(TensorCPUTest, SliceF16) {
  Tensor<bfloat16, CPU> tensor({1, 4});
  Tensor<bfloat16, CPU> exp({1, 2});

  tensor.fill_(bfloat16(2.0));
  exp.fill_(bfloat16(2.0));

  auto view = tensor.view();

  Tensor<bfloat16, CPU> result = slice(view, -1, 0, 2);

  fmt::println("result, {}", result.view());

  Shape expected_shape = {1, 2};

  EXPECT_EQ(result.shape(), expected_shape);

  tensor_is_close<bfloat16>(result.span(), exp.span());
}

TEST(TensorCPUTest, TrilBF16) {
  Tensor<bfloat16, CPU> tensor({4, 4});
  tensor.fill_(1.0);

  Tensor<bfloat16, CPU> no_diag = tril<bfloat16>(tensor.view(), false);

  std::vector<bfloat16> exp = {1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1};

  tensor_is_close<bfloat16>(no_diag.span(), std::span(exp));
  Tensor<bfloat16, CPU> diag = tril<bfloat16>(tensor.view(), true);

  exp = {1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  tensor_is_close<bfloat16>(diag.span(), std::span(exp));
}

TEST(TensorCPUTest, ArgmaxInt) {
  Tensor<int, CPU> tensor({4, 4});
  tensor.fill_(1);
  tensor.set_(2, 4); // idx 2 of first batch element
  tensor.set_(7, 8); // idx 3 of second batch element

  auto maxes = argmax(tensor.view(), -1, true);

  fmt::println("MAXES {}", maxes.view());

  std::vector<int> exp{2, 3, 0, 0};
  tensor_is_close<int>(maxes.span(), std::span(exp));
}
