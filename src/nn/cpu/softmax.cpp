#include <nn/softmax.hpp>
#include <tensor/ops.hpp>

using namespace nn;
using namespace tensor;

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> Softmax::operator()(const TensorView<T, D>& input,
                                                      int dim) const {
  Tensor<float, D> f32 = input.template to<float>();

  auto maxes = max(f32.view(), dim, true);

  auto scaled = sub(f32.view(), maxes.view());

  auto expd = scaled.view().exp();

  auto expd_sum = sum(expd.view(), dim, true);

  auto out = div(expd.view(), expd_sum.view());

  return out.view().template to<T>();
}

template Tensor<bfloat16, CPU> Softmax::operator()(const TensorView<bfloat16, CPU>& input,
                                                   int dim) const;
