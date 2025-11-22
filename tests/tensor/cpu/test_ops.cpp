#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <fmt/format.h>

#include <tensor/ops.hpp>

using namespace tensor;

TEST(TensorCPUTest, AddInt) {
  Tensor<int, CPU> a({2, 4});
  Tensor<int, CPU> b({2, 4});
  Tensor<int, CPU> exp({2, 4});

  a.fill_(4);
  b.fill_(3);
  exp.fill_(7);

  auto a_v = a.view();
  auto b_v = b.view();

  Tensor<int, CPU> result = add(a_v, b_v);

  tensor_is_close<int>(result.span(), exp.span());
}

TEST(TensorCPUTest, AddBF16) {
  Tensor<bfloat16, CPU> a({2, 4});
  Tensor<bfloat16, CPU> b({2, 4});
  Tensor<bfloat16, CPU> exp({2, 4});

  a.fill_(bfloat16(4.0));
  b.fill_(bfloat16(3.0));
  exp.fill_(bfloat16(7.0));

  auto a_v = a.view();
  auto b_v = b.view();

  Tensor<bfloat16, CPU> result = add(a_v, b_v);

  tensor_is_close<bfloat16>(result.span(), exp.span());
}
