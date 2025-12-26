#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/ops.hpp>

using namespace tensor;

TEST(TensorCUDATest, Arange) {
  SKIP_IF_NO_GPU();
  Tensor<int, CUDA> result = arange<int, CUDA>(0, 10, 2);

  auto cpu = result.cpu();

  std::vector<int> exp = {0, 2, 4, 6, 8};

  tensor_is_close<int>(cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, AddBF16) {
  SKIP_IF_NO_GPU();
  Tensor<bfloat16, CUDA> tensor_a({16384, 2048});
  Tensor<bfloat16, CUDA> tensor_b({16384, 2048});

  Tensor<bfloat16, CPU> exp({16384, 2048});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));
  exp.fill_(bfloat16(7.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<bfloat16, CUDA> result = add(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, SubFp32) {
  SKIP_IF_NO_GPU();
  Tensor<float, CUDA> tensor_a({16384, 2048});
  Tensor<float, CUDA> tensor_b({16384, 2048});

  Tensor<float, CPU> exp({16384, 2048});

  tensor_a.fill_(float(4.0));
  tensor_b.fill_(float(3.0));
  exp.fill_(float(1.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<float, CUDA> result = sub(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<float>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, DivFp32) {
  SKIP_IF_NO_GPU();
  Tensor<float, CUDA> tensor_a({16384, 2048});
  Tensor<float, CUDA> tensor_b({16384, 2048});

  Tensor<float, CPU> exp({16384, 2048});

  tensor_a.fill_(float(8.0));
  tensor_b.fill_(float(4.0));
  exp.fill_(float(2.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<float, CUDA> result = div(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<float>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, DivScalarFp32) {
  SKIP_IF_NO_GPU();
  Tensor<float, CUDA> tensor_a({16384, 2048});

  Tensor<float, CPU> exp({16384, 2048});

  tensor_a.fill_(float(8.0));
  exp.fill_(float(2.0));

  auto a_v = tensor_a.view();

  Tensor<float, CUDA> result = div(a_v, 4.0);

  auto cpu = result.cpu();

  tensor_is_close<float>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, MulBf16) {
  SKIP_IF_NO_GPU();
  Tensor<bfloat16, CUDA> tensor_a({16384, 2048});
  Tensor<bfloat16, CUDA> tensor_b({16384, 2048});

  Tensor<bfloat16, CPU> exp({16384, 2048});

  tensor_a.fill_(bfloat16(8.0));
  tensor_b.fill_(bfloat16(4.0));
  exp.fill_(bfloat16(32.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  Tensor<bfloat16, CUDA> result = mul(a_v, b_v);

  auto cpu = result.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, MulScalarBf16) {
  SKIP_IF_NO_GPU();
  Tensor<bfloat16, CUDA> tensor_a({16384, 2048});

  Tensor<bfloat16, CPU> exp({16384, 2048});

  tensor_a.fill_(bfloat16(8.0));
  exp.fill_(bfloat16(32.0));

  auto a_v = tensor_a.view();

  Tensor<bfloat16, CUDA> result = mul(a_v, 4.0);

  auto cpu = result.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}
