#include <tensor/ops.hpp>

namespace tensor {

// element-wise add

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

// element-wise mul

template <DType T, Device D>
Tensor<T, D> mul(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b) {
  Tensor<T, D> out{tensor_a.shape};
  for (int i = 0; i < out.size(); ++i) {
    out.span()[i] = tensor_a.span()[i] * tensor_b.span()[i];
  }
  return out;
}

template Tensor<bfloat16, CPU> mul(TensorView<bfloat16, CPU>, TensorView<bfloat16, CPU>);

// matmul

template <DType T, Device D>
Tensor<T, D> matmul(TensorView<T, D> tensor_a, TensorView<T, D> tensor_b) {
  // a: [..., M, K]
  // b: [K, N] (2D) or [..., K, N] (batched)
  // out: [..., M, N]

  size_t a_ndim = tensor_a.shape.size();
  size_t b_ndim = tensor_b.shape.size();

  assert(a_ndim >= 2 && b_ndim >= 2);

  size_t M = tensor_a.shape[a_ndim - 2]; // NOLINT
  size_t K = tensor_a.shape[a_ndim - 1]; // NOLINT
  size_t N = tensor_b.shape[b_ndim - 1]; // NOLINT

  assert(K == tensor_b.shape[b_ndim - 2] && "Inner dimensions must match");

  size_t a_stride_row = tensor_a.stride[a_ndim - 2];
  size_t a_stride_col = tensor_a.stride[a_ndim - 1];
  size_t b_stride_row = tensor_b.stride[b_ndim - 2];
  size_t b_stride_col = tensor_b.stride[b_ndim - 1];

  size_t batch_size = 1;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    batch_size *= tensor_a.shape[i];
  }

  size_t a_batch_stride = (a_ndim > 2) ? tensor_a.stride[a_ndim - 3] : 0;
  size_t b_batch_stride = (b_ndim > 2) ? tensor_b.stride[b_ndim - 3] : 0;

  Shape out_shape;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    out_shape.push_back(tensor_a.shape[i]);
  }
  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor<T, D> out{out_shape};

  auto a_data = tensor_a.span();
  auto b_data = tensor_b.span();
  auto out_data = out.span();

  // Output strides (contiguous)
  size_t out_batch_stride = M * N;
  size_t out_stride_row = N;

  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t a_batch_offset = batch * a_batch_stride;
    size_t b_batch_offset = batch * b_batch_stride; // 0 for each batch if b is 2D
    size_t out_batch_offset = batch * out_batch_stride;

    for (size_t row = 0; row < M; ++row) {
      for (size_t col = 0; col < N; ++col) {
        float sum = 0.0;

        for (size_t i = 0; i < K; ++i) {
          size_t a_idx = a_batch_offset + (row * a_stride_row) + (i * a_stride_col);
          size_t b_idx = b_batch_offset + (i * b_stride_row) + (col * b_stride_col);
          sum += static_cast<float>(a_data[a_idx]) * static_cast<float>(b_data[b_idx]);
        }

        size_t out_idx = out_batch_offset + (row * out_stride_row) + col;
        out_data[out_idx] = static_cast<T>(sum);
      }
    }
  }

  return out;
}

template Tensor<bfloat16, CPU> matmul(TensorView<bfloat16, CPU>, TensorView<bfloat16, CPU>);

} // namespace tensor
