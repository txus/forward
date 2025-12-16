#include <nn/softmax.hpp>
#include <tensor/ops.hpp>

using namespace nn;
using namespace tensor;

template <DType T, Device D>
Tensor<T, D> Softmax::operator()(const TensorView<T, D>& input, int dim) const {
  Tensor<T, D> out{{input.shape}};

  auto maxes = max(input, dim, true);

  auto scaled = sub(input, maxes.view());

  auto expd = scaled.view().exp();

  auto expd_sum = sum(expd.view(), dim, true);

  return div(expd.view(), expd_sum.view());
}

template Tensor<bfloat16, CPU> Softmax::operator()(const TensorView<bfloat16, CPU>& input,
                                                   int dim) const;
