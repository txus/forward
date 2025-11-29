#include <tensor/ops.hpp>

namespace tensor {

template <DType T, Device D>
Tensor<T, D> add(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b) {
  Tensor<T, D> out{tensor_a.shape};
  for (int i = 0; i < out.size(); ++i) {
    out.span()[i] = tensor_a.span()[i] + tensor_b.span()[i];
  }
  return out;
}

template Tensor<bfloat16, CPU> add(TensorView<bfloat16, CPU>, TensorView<bfloat16, CPU>);
template Tensor<int, CPU> add(TensorView<int, CPU>, TensorView<int, CPU>);

template <DType T, Device D>
Tensor<T, D> matmul(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b) {
  assert(tensor_a.shape[1] == tensor_b.shape[0]);

  auto m = tensor_a.shape[0]; // NOLINT
  auto k = tensor_a.shape[1]; // NOLINT
  auto n = tensor_b.shape[1]; // NOLINT

  Tensor<T, D> out{{m, n}};

  auto a_span = tensor_a.span(); // a is MxK
  auto b_span = tensor_b.span(); // b is KxN
  auto out_span = out.span();    // out is MxN

  for (size_t row_idx = 0; row_idx < m; ++row_idx) {
    for (size_t col_idx = 0; col_idx < n; ++col_idx) {
      float sum = 0.0; // accumulate in fp32
      for (size_t off = 0; off < k; ++off) {
        sum += a_span[(row_idx * k) + off] * b_span[(off * n) + col_idx];
      }
      out_span[(row_idx * m) + col_idx] = sum;
    }
  }
  return out;
}

template Tensor<bfloat16, CPU> matmul(TensorView<bfloat16, CPU>, TensorView<bfloat16, CPU>);

} // namespace tensor
