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

TEST(TensorCUDATest, AddBf16Broadcasting) {
  SKIP_IF_NO_GPU();
  Tensor<bfloat16, CUDA> tensor_a({16384, 2048});
  Tensor<bfloat16, CUDA> tensor_b({16384, 1});

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

TEST(TensorCUDATest, CatBf16LastDim) {
  Tensor<bfloat16, CUDA> tensor_a({2, 4});
  Tensor<bfloat16, CUDA> tensor_b({2, 2});

  tensor_a.fill_(bfloat16(2.0));
  tensor_b.fill_(bfloat16(3.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  std::vector<bfloat16> exp = {2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0};

  Tensor<bfloat16, CUDA> result = cat(a_v, b_v, 1);

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 6};

  EXPECT_EQ(result_cpu.shape(), expected_shape);

  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, CatBf16FirstDim) {
  Tensor<bfloat16, CUDA> tensor_a({3, 2});
  Tensor<bfloat16, CUDA> tensor_b({2, 2});

  tensor_a.fill_(bfloat16(2.0));
  tensor_b.fill_(bfloat16(3.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  std::vector<bfloat16> exp = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0};

  Tensor<bfloat16, CUDA> result = cat(a_v, b_v, 0);

  auto result_cpu = result.cpu();

  Shape expected_shape = {5, 2};

  EXPECT_EQ(result_cpu.shape(), expected_shape);

  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, TrilBf16) {
  Tensor<bfloat16, CUDA> tensor({4, 4});
  tensor.fill_(1.0);

  Tensor<bfloat16, CUDA> no_diag_ = tril<bfloat16>(tensor.view(), false);

  auto no_diag = no_diag_.cpu();

  fmt::println("no diag: {}", no_diag.view());

  std::vector<bfloat16> exp = {1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1};

  tensor_is_close<bfloat16>(no_diag.span(), std::span(exp));
  Tensor<bfloat16, CUDA> diag_ = tril<bfloat16>(tensor.view(), true);

  auto diag = diag_.cpu();
  fmt::println("diag: {}", diag.view());

  exp = {1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  tensor_is_close<bfloat16>(diag.span(), std::span(exp));
}

TEST(TensorCUDATest, SliceBf16FirstDim) {
  SKIP_IF_NO_GPU();
  // Tensor shape {4, 3}: 4 rows, 3 cols
  // Data: row0=[1,2,3], row1=[4,5,6], row2=[7,8,9], row3=[10,11,12]
  Tensor<bfloat16, CPU> tensor({4, 3});
  for (int i = 0; i < 12; ++i) {
    tensor.set_(i, bfloat16(i + 1));
  }

  auto gpu_tensor = tensor.cuda();

  // Slice rows 1 to 3 (exclusive), so rows 1 and 2
  Tensor<bfloat16, CUDA> result = slice(gpu_tensor.view(), 0, 1, 3);

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 3};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  // Expected: row1=[4,5,6], row2=[7,8,9]
  std::vector<bfloat16> exp = {4, 5, 6, 7, 8, 9};
  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, SliceBf16LastDim) {
  SKIP_IF_NO_GPU();
  // Tensor shape {2, 6}
  // Data: row0=[1,2,3,4,5,6], row1=[7,8,9,10,11,12]
  Tensor<bfloat16, CPU> tensor({2, 6});
  for (int i = 0; i < 12; ++i) {
    tensor.set_(i, bfloat16(i + 1));
  }

  auto gpu_tensor = tensor.cuda();

  // Slice cols 2 to 5 (exclusive), so cols 2, 3, 4
  Tensor<bfloat16, CUDA> result = slice(gpu_tensor.view(), 1, 2, 5);

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 3};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  // Expected: row0=[3,4,5], row1=[9,10,11]
  std::vector<bfloat16> exp = {3, 4, 5, 9, 10, 11};
  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, SliceBf16MiddleDim) {
  SKIP_IF_NO_GPU();
  // Tensor shape {2, 4, 3}: 2 batches, 4 rows, 3 cols
  Tensor<bfloat16, CPU> tensor({2, 4, 3});
  for (int i = 0; i < 24; ++i) {
    tensor.set_(i, bfloat16(i + 1));
  }

  auto gpu_tensor = tensor.cuda();

  // Slice dim 1 (rows) from 1 to 3, keeping 2 rows
  Tensor<bfloat16, CUDA> result = slice(gpu_tensor.view(), 1, 1, 3);

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 2, 3};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  // Batch 0: rows 1-2 = [4,5,6, 7,8,9]
  // Batch 1: rows 1-2 = [16,17,18, 19,20,21]
  std::vector<bfloat16> exp = {4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21};
  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, MatmulBf16) {
  SKIP_IF_NO_GPU();
  // A: 2x3 matrix
  // [[1, 2, 3],
  //  [4, 5, 6]]
  Tensor<bfloat16, CPU> a({2, 3});
  a.set_(0, 1);
  a.set_(1, 2);
  a.set_(2, 3);
  a.set_(3, 4);
  a.set_(4, 5);
  a.set_(5, 6);

  // B: 3x2 matrix
  // [[7, 8],
  //  [9, 10],
  //  [11, 12]]
  Tensor<bfloat16, CPU> b({3, 2});
  b.set_(0, 7);
  b.set_(1, 8);
  b.set_(2, 9);
  b.set_(3, 10);
  b.set_(4, 11);
  b.set_(5, 12);

  auto a_gpu = a.cuda();
  auto b_gpu = b.cuda();

  // C = A @ B should be 2x2
  // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
  // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
  // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
  // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
  Tensor<bfloat16, CUDA> result = matmul(a_gpu.view(), b_gpu.view());

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 2};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  std::vector<bfloat16> exp = {58, 64, 139, 154};
  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, MatmulBf16Batched) {
  SKIP_IF_NO_GPU();
  // Batched matmul: 2 batches of 2x3 @ 3x2
  Tensor<bfloat16, CPU> a({2, 2, 3});
  // Batch 0: same as above test
  a.set_(0, 1);
  a.set_(1, 2);
  a.set_(2, 3);
  a.set_(3, 4);
  a.set_(4, 5);
  a.set_(5, 6);
  // Batch 1: all ones
  for (int i = 6; i < 12; ++i) {
    a.set_(i, 1);
  }

  Tensor<bfloat16, CPU> b({2, 3, 2});
  // Batch 0: same as above test
  b.set_(0, 7);
  b.set_(1, 8);
  b.set_(2, 9);
  b.set_(3, 10);
  b.set_(4, 11);
  b.set_(5, 12);
  // Batch 1: all twos
  for (int i = 6; i < 12; ++i) {
    b.set_(i, 2);
  }

  auto a_gpu = a.cuda();
  auto b_gpu = b.cuda();

  Tensor<bfloat16, CUDA> result = matmul(a_gpu.view(), b_gpu.view());

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 2, 2};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  // Batch 0: same as single matmul test
  // Batch 1: all ones @ all twos = each element is 3*2 = 6
  std::vector<bfloat16> exp = {58, 64, 139, 154, 6, 6, 6, 6};
  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, MatmulFp32) {
  SKIP_IF_NO_GPU();
  Tensor<float, CPU> a({2, 3});
  a.set_(0, 1);
  a.set_(1, 2);
  a.set_(2, 3);
  a.set_(3, 4);
  a.set_(4, 5);
  a.set_(5, 6);

  Tensor<float, CPU> b({3, 2});
  b.set_(0, 7);
  b.set_(1, 8);
  b.set_(2, 9);
  b.set_(3, 10);
  b.set_(4, 11);
  b.set_(5, 12);

  auto a_gpu = a.cuda();
  auto b_gpu = b.cuda();

  Tensor<float, CUDA> result = matmul(a_gpu.view(), b_gpu.view());

  auto result_cpu = result.cpu();

  Shape expected_shape = {2, 2};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  std::vector<float> exp = {58, 64, 139, 154};
  tensor_is_close<float>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, Copy) {
  tensor::Tensor<int, tensor::CUDA> tensor({2, 4});
  tensor.fill_(6);

  auto view = tensor.view();

  auto new_t = copy(view);

  EXPECT_EQ(tensor.at(0), 6);
  EXPECT_EQ(new_t.at(0), 6);
}

TEST(TensorCUDATest, CopySlice) {
  tensor::Tensor<int, tensor::CUDA> tensor({2, 4});
  tensor.fill_(6);

  auto view = tensor.view().get(0);

  auto new_t = copy(view);

  EXPECT_EQ(tensor.at(0), 6);
  EXPECT_EQ(new_t.at(0), 6);
}
