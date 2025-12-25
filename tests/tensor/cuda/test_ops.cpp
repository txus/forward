#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/ops.hpp>

using namespace tensor;

TEST(TensorCUDATest, Arange) {
  Tensor<int, CUDA> result = arange<int, CUDA>(0, 10, 2);

  auto cpu = result.cpu();

  std::vector<int> exp = {0, 2, 4, 6, 8};

  tensor_is_close<int>(cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, AddInt) {
  Tensor<int, CUDA> tensor_a({2, 4});
  Tensor<int, CUDA> tensor_b({2, 4});

  Tensor<int, CPU> exp({2, 4});

  tensor_a.fill_(4);
  tensor_b.fill_(3);
  exp.fill_(7);

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<int, CUDA> result = add(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<int>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, AddBF16) {
  Tensor<bfloat16, CUDA> tensor_a({2, 4});
  Tensor<bfloat16, CUDA> tensor_b({2, 4});

  Tensor<bfloat16, CPU> exp({2, 4});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));
  exp.fill_(bfloat16(7.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<bfloat16, CUDA> result = add(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, SubBF16) {
  Tensor<bfloat16, CUDA> tensor_a({2, 4});
  Tensor<bfloat16, CUDA> tensor_b({2, 4});

  Tensor<bfloat16, CPU> exp({2, 4});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));
  exp.fill_(bfloat16(1.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<bfloat16, CUDA> result = sub(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, SubScalarBF16) {
  Tensor<bfloat16, CUDA> tensor({2, 4});
  bfloat16 scalar{3.0};

  Tensor<bfloat16, CPU> exp({2, 4});

  tensor.fill_(bfloat16(4.0));
  exp.fill_(bfloat16(1.0));

  auto a_v = tensor.view();

  Tensor<bfloat16, CUDA> result = sub(a_v, scalar);

  auto cpu = result.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}
