#include <tensor/ops.hpp>

namespace tensor {

template <DType T, Device D>
Tensor<T, D> add(TensorView<T, D>& tensor_a, TensorView<T, D>& tensor_b) {
  Tensor<T, D> out{tensor_a.shape};
  for (int i = 0; i < out.size(); ++i) {
    out.span()[i] = tensor_a.span()[i] + tensor_b.span()[i];
  }
  return out;
}

template Tensor<bfloat16, CPU> add(TensorView<bfloat16, CPU>&, TensorView<bfloat16, CPU>&);
template Tensor<int, CPU> add(TensorView<int, CPU>&, TensorView<int, CPU>&);

} // namespace tensor
