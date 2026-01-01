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

TEST(TensorCUDATest, MulBf16BroadcastMiddleDim) {
  SKIP_IF_NO_GPU();
  // Test broadcasting on middle dimension (like attention heads in GQA)
  // a: {1, 4, 2, 2} - e.g., batch=1, heads=4, seq=2, dim=2
  // b: {1, 1, 2, 2} - broadcasts across heads
  Tensor<bfloat16, CPU> tensor_a_cpu({1, 4, 2, 2});
  Tensor<bfloat16, CPU> tensor_b_cpu({1, 1, 2, 2});

  // Fill a with values 1-16
  for (int i = 0; i < 16; ++i) {
    tensor_a_cpu.set_(i, bfloat16(i + 1));
  }

  // Fill b with [2, 3, 4, 5]
  tensor_b_cpu.set_(0, bfloat16(2.0));
  tensor_b_cpu.set_(1, bfloat16(3.0));
  tensor_b_cpu.set_(2, bfloat16(4.0));
  tensor_b_cpu.set_(3, bfloat16(5.0));

  auto tensor_a = tensor_a_cpu.cuda();
  auto tensor_b = tensor_b_cpu.cuda();

  Tensor<bfloat16, CUDA> result = mul(tensor_a.view(), tensor_b.view());

  auto result_cpu = result.cpu();

  // Verify shape
  Shape expected_shape = {1, 4, 2, 2};
  EXPECT_EQ(result_cpu.shape(), expected_shape);

  // Each head should have same pattern: element-wise multiply with b
  // Head 0: [1,2,3,4] * [2,3,4,5] = [2,6,12,20]
  // Head 1: [5,6,7,8] * [2,3,4,5] = [10,18,28,40]
  // Head 2: [9,10,11,12] * [2,3,4,5] = [18,30,44,60]
  // Head 3: [13,14,15,16] * [2,3,4,5] = [26,42,60,80]
  std::vector<bfloat16> expected = {2,  6,  12, 20, 10, 18, 28, 40,
                                     18, 30, 44, 60, 26, 42, 60, 80};

  tensor_is_close<bfloat16>(result_cpu.span(), std::span(expected));
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
  SKIP_IF_NO_GPU();
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

  tensor_is_close<bfloat16>(result_cpu.span(), std::span(exp));
}

TEST(TensorCUDATest, MaskedFillBf16Broadcasting) {
  SKIP_IF_NO_GPU();
  // Input shape: {1, 4, 2, 2} - e.g., batch=1, heads=4, seq_len=2, key_len=2
  // Mask shape: {1, 1, 2, 2} - broadcasts across heads dimension
  // This tests the case used in GQA where attention mask broadcasts over heads

  Tensor<bfloat16, CPU> tensor({1, 4, 2, 2});
  tensor.fill_(bfloat16(1.0));

  // Causal mask: lower triangle = 1 (keep), upper triangle = 0 (mask out)
  // [1, 0]
  // [1, 1]
  Tensor<int, CPU> mask({1, 1, 2, 2});
  mask.set_(0, 1); // [0,0] = keep
  mask.set_(1, 0); // [0,1] = mask
  mask.set_(2, 1); // [1,0] = keep
  mask.set_(3, 1); // [1,1] = keep

  auto tensor_gpu = tensor.cuda();
  auto mask_gpu = mask.cuda();

  bfloat16 neg_inf = std::numeric_limits<bfloat16>::lowest();
  Tensor<bfloat16, CUDA> result = masked_fill(tensor_gpu.view(), mask_gpu.view(), neg_inf);

  auto result_cpu = result.cpu();

  // All 4 heads should have same pattern: [1, neg_inf, 1, 1]
  for (size_t head = 0; head < 4; ++head) {
    size_t base = head * 4;
    EXPECT_FLOAT_EQ(float(result_cpu.at(base + 0)), 1.0f) << "head " << head << " [0,0]";
    EXPECT_FLOAT_EQ(float(result_cpu.at(base + 1)), float(neg_inf)) << "head " << head << " [0,1]";
    EXPECT_FLOAT_EQ(float(result_cpu.at(base + 2)), 1.0f) << "head " << head << " [1,0]";
    EXPECT_FLOAT_EQ(float(result_cpu.at(base + 3)), 1.0f) << "head " << head << " [1,1]";
  }
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

TEST(TensorCUDATest, CopyTransposed) {
  // Create a 2x3 matrix on CPU, copy to GPU, transpose, and copy to materialize
  // Original: [[1,2,3], [4,5,6]]
  Tensor<int, CPU> cpu_tensor({2, 3});
  cpu_tensor.set_(0, 1);
  cpu_tensor.set_(1, 2);
  cpu_tensor.set_(2, 3);
  cpu_tensor.set_(3, 4);
  cpu_tensor.set_(4, 5);
  cpu_tensor.set_(5, 6);

  auto gpu_tensor = cpu_tensor.cuda();

  // Transpose to 3x2 - this makes it non-contiguous
  auto transposed = gpu_tensor.view();
  transposed.transpose(0, 1);

  EXPECT_FALSE(transposed.is_contiguous());

  // Copy should materialize the transposed view into contiguous memory
  auto materialized = copy(transposed);

  EXPECT_TRUE(materialized.view().is_contiguous());
  EXPECT_EQ(materialized.shape()[0], 3);
  EXPECT_EQ(materialized.shape()[1], 2);

  // Verify values are correctly transposed
  auto result_cpu = materialized.cpu();
  // Original: [[1,2,3], [4,5,6]]
  // Transposed: [[1,4], [2,5], [3,6]]
  std::vector<int> expected = {1, 4, 2, 5, 3, 6};
  tensor_is_close<int>(result_cpu.span(), std::span(expected));
}

TEST(TensorCUDATest, CopyTransposed4D) {
  // Create a 4D tensor like in GQA: {batch=1, num_heads=2, seq_len=3, head_dim=2}
  // After transpose(1,2): {1, 3, 2, 2}
  Tensor<int, CPU> cpu_tensor({1, 2, 3, 2});
  // Fill with sequential values: 0, 1, 2, ..., 11
  for (int i = 0; i < 12; ++i) {
    cpu_tensor.set_(i, i + 1);
  }

  auto gpu_tensor = cpu_tensor.cuda();

  // Transpose dims 1 and 2: {1, 2, 3, 2} -> {1, 3, 2, 2}
  auto transposed = gpu_tensor.view();
  transposed.transpose(1, 2);

  EXPECT_FALSE(transposed.is_contiguous());
  EXPECT_EQ(transposed.shape[0], 1);
  EXPECT_EQ(transposed.shape[1], 3);
  EXPECT_EQ(transposed.shape[2], 2);
  EXPECT_EQ(transposed.shape[3], 2);

  // Copy should materialize the transposed view
  auto materialized = copy(transposed);

  EXPECT_TRUE(materialized.view().is_contiguous());

  // Verify values
  auto result_cpu = materialized.cpu();
  // Original layout (1,2,3,2) in memory: head0_seq0, head0_seq1, head0_seq2, head1_seq0, head1_seq1, head1_seq2
  // [[[[ 1, 2], [ 3, 4], [ 5, 6]],   <- head 0, seq 0,1,2
  //   [[ 7, 8], [ 9,10], [11,12]]]]  <- head 1, seq 0,1,2
  // After transpose(1,2) to (1,3,2,2):
  // [[[[ 1, 2], [ 7, 8]],   <- seq 0, head 0,1
  //   [[ 3, 4], [ 9,10]],   <- seq 1, head 0,1
  //   [[ 5, 6], [11,12]]]]  <- seq 2, head 0,1
  std::vector<int> expected = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  tensor_is_close<int>(result_cpu.span(), std::span(expected));
}
