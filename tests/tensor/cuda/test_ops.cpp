#include <fmt/format.h>
#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/ops.hpp>

using namespace tensor;

TEST(TensorCUDATest, ArgmaxBf16) {
  SKIP_IF_NO_GPU();
  Tensor<bfloat16, CPU> tensor({4, 4});
  tensor.fill_(1.0);
  tensor.set_(2., 4.0); // idx 2 of first batch element
  tensor.set_(7, 8.0);  // idx 3 of second batch element

  Tensor<bfloat16, CUDA> input = tensor.cuda();

  auto maxes = argmax(input.view(), -1, true);

  auto cpu = maxes.cpu();

  std::vector<int> exp{2, 3, 0, 0};
  tensor_is_close<int>(cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, Arange) {
  SKIP_IF_NO_GPU();
  Tensor<int, CUDA> result = arange<int, CUDA>(0, 10, 2);

  auto cpu = result.cpu();

  std::vector<int> exp = {0, 2, 4, 6, 8};

  tensor_is_close<int>(cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, ReplaceFromBf16) {
  SKIP_IF_NO_GPU();
  Tensor<bfloat16, CUDA> input({16384, 2048});
  input.fill_(bfloat16(4.0));

  Tensor<bfloat16, CUDA> output({16384, 2048});

  Tensor<bfloat16, CPU> exp({16384, 2048});
  exp.fill_(bfloat16(4.0));

  auto input_v = input.view();

  replace_from_(output, input_v);

  auto cpu = output.cpu();

  tensor_is_close<bfloat16>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, AddBf16) {
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

TEST(TensorCUDATest, SumFp32) {
  SKIP_IF_NO_GPU();
  Tensor<float, CUDA> tensor({16384, 2048});

  Tensor<float, CPU> exp({16384, 1});

  tensor.fill_(float(1.0));
  exp.fill_(float(2048.0));

  auto view = tensor.view();

  Tensor<float, CUDA> result = sum(view, 1, true);

  auto cpu = result.cpu();

  tensor_is_close<float>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, MaxFp32) {
  SKIP_IF_NO_GPU();
  Tensor<float, CPU> tensor({16384, 2048});
  tensor.fill_(float(1.0));

  tensor.set_(4, 8.0);
  tensor.set_(2049, 9.0);
  tensor.set_(4099, 10.0);

  auto cuda_tensor = tensor.cuda();

  Tensor<float, CPU> exp({16384, 1});

  exp.fill_(float(1.0));
  exp.set_(0, 8.0);
  exp.set_(1, 9.0);
  exp.set_(2, 10.0);

  auto view = tensor.view();

  Tensor<float, CUDA> result = max(cuda_tensor.view(), 1, true);

  auto cpu = result.cpu();

  tensor_is_close<float>(cpu.span(), exp.span());
}

TEST(TensorCUDATest, MaskedFillBf16) {
  Tensor<bfloat16, CPU> tensor({3, 4});
  tensor.fill_(bfloat16(4.0));

  Tensor<int, CPU> mask({4});
  mask.set_(0, 0);
  mask.set_(1, 1);
  mask.set_(2, 0);
  mask.set_(3, 1);

  auto tensor_gpu = tensor.cuda();
  auto mask_gpu = mask.cuda();

  std::vector<bfloat16> exp = {0, 4.0, 0, 4.0, 0, 4.0, 0, 4.0, 0, 4.0, 0, 4.0};

  Tensor<bfloat16, CUDA> result = masked_fill(tensor_gpu.view(), mask_gpu.view(), 0.0);

  auto result_cpu = result.cpu();

  fmt::println("result {}", result_cpu.view());

  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}
