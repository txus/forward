#include <tensor/ops.hpp>

namespace tensor {

template <DType T, Device D>
Tensor<T, D> add(TensorView<T, D> &a, TensorView<T, D> &b) {
  Tensor<T, D> out{a.shape};
  for (int i = 0; i < out.size(); ++i) {
    out.span()[i] = a.span()[i] + b.span()[i];
  }
  return out;
}

template Tensor<bfloat16, CPU> add(TensorView<bfloat16, CPU> &,
                                   TensorView<bfloat16, CPU> &);
template Tensor<int, CPU> add(TensorView<int, CPU> &, TensorView<int, CPU> &);

} // namespace tensor
