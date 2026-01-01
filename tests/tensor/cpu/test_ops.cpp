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

TEST(TensorCPUTest, AddBf16) {
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

TEST(TensorCPUTest, MaxFp32) {
  Tensor<float, CPU> tensor({16384, 2048});
  tensor.fill_(float(1.0));

  tensor.set_(4, 8.0);
  tensor.set_(2049, 9.0);
  tensor.set_(4099, 10.0);

  Tensor<float, CPU> exp({16384, 1});

  exp.fill_(float(1.0));
  exp.set_(0, 8.0);
  exp.set_(1, 9.0);
  exp.set_(2, 10.0);

  auto view = tensor.view();

  Tensor<float, CPU> result = max(tensor.view(), 1, true);

  tensor_is_close<float>(result.span(), exp.span());
}

TEST(TensorCPUTest, PowBf16) {
  Tensor<bfloat16, CPU> tensor({2, 2});
  tensor.fill_(2.0);

  Tensor<bfloat16, CPU> result = pow<bfloat16>(3.0, tensor.view());

  std::vector<bfloat16> exp = {9, 9, 9, 9};

  tensor_is_close<bfloat16>(result.span(), std::span(exp));
}

TEST(TensorCPUTest, MatmulBf16) {
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

TEST(TensorCPUTest, CatFp16) {
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

TEST(TensorCPUTest, SliceFp16) {
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

TEST(TensorCPUTest, TrilBf16) {
  Tensor<bfloat16, CPU> tensor({4, 4});
  tensor.fill_(1.0);

  Tensor<bfloat16, CPU> no_diag = tril<bfloat16>(tensor.view(), false);

  std::vector<bfloat16> exp = {1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1};

  tensor_is_close<bfloat16>(no_diag.span(), std::span(exp));
  Tensor<bfloat16, CPU> diag = tril<bfloat16>(tensor.view(), true);

  exp = {1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  tensor_is_close<bfloat16>(diag.span(), std::span(exp));
}

TEST(TensorCPUTest, ArgmaxBf16) {
  Tensor<bfloat16, CPU> tensor({4, 4});
  tensor.fill_(1.0);
  tensor.set_(2., 4.0); // idx 2 of first batch element
  tensor.set_(7, 8.0);  // idx 3 of second batch element

  auto maxes = argmax(tensor.view(), -1, true);

  std::vector<int> exp{2, 3, 0, 0};
  tensor_is_close<int>(maxes.span(), std::span(exp));
}

TEST(TensorCPUTest, MaskedFillBf16) {
  Tensor<bfloat16, CPU> tensor({3, 4});
  tensor.fill_(bfloat16(4.0));

  Tensor<int, CPU> mask({4});
  mask.set_(0, 0);
  mask.set_(1, 1);
  mask.set_(2, 0);
  mask.set_(3, 1);

  std::vector<bfloat16> exp = {0, 4.0, 0, 4.0, 0, 4.0, 0, 4.0, 0, 4.0, 0, 4.0};

  auto view = tensor.view();
  auto mask_view = mask.view();

  Tensor<bfloat16, CPU> result = masked_fill(view, mask_view, 0.0);

  fmt::println("result {}", result.view());

  tensor_is_close<bfloat16>(result.span(), std::span(exp));
}

TEST(TensorCPUTest, Copy) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  auto view = tensor.view();

  auto new_t = copy(view);
  new_t.set_(0, 6);
  tensor.set_(0, 7);

  EXPECT_EQ(tensor.at(0), 7);
  EXPECT_EQ(new_t.at(0), 6);
}

TEST(TensorCPUTest, CopySlice) {
  tensor::Tensor<int, tensor::CPU> tensor({2, 4});

  auto view = tensor.view().get(0);

  auto new_t = copy(view);
  new_t.set_(0, 6);
  tensor.set_(0, 7);

  EXPECT_EQ(tensor.at(0), 7);
  EXPECT_EQ(new_t.at(0), 6);
}
