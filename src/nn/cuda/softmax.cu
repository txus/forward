#include <nn/softmax.hpp>
#include <tensor/ops.hpp>

using namespace nn;
using namespace tensor;

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> Softmax::operator()(const TensorView<T, D>& input,
                                                      int dim) const {
  Tensor<float, D> f32 = to<T, float>(input);
  auto maxes = tensor::max(f32.view(), dim, true);
  auto scaled = sub(f32.view(), maxes.view());
  auto expd = tensor::exp(scaled.view());
  auto expd_sum = sum(expd.view(), dim, true);
  auto out = tensor::div(expd.view(), expd_sum.view());

  return to<float, T>(out.view());
}

template Tensor<bfloat16, CUDA> Softmax::operator()(const TensorView<bfloat16, CUDA>& input,
                                                   int dim) const;
