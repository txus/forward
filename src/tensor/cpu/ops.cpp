#include <tensor/ops.hpp>

namespace tensor {

// constructors

template <DType T, Device D> Tensor<T, D> arange(T start, T end, T step) {
  std::vector<T> vec{};

  for (T element = start; element < end; element += step) {
    vec.push_back(element);
  }

  return Tensor<T, D>{Shape{vec.size()}, std::move(vec)};
}

template Tensor<int, CPU> arange(int start, int end, int step = 1);
template Tensor<float, CPU> arange(float start, float end, float step = 1);
template Tensor<bfloat16, CPU> arange(bfloat16 start, bfloat16 end, bfloat16 step = 1);

// element-wise ops

template <DType T, Device D, typename Func>
Tensor<T, D> element_wise(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b,
                          Func func) {
  Shape out_shape = broadcast_shape(tensor_a.shape, tensor_b.shape);
  Shape a_strides = broadcast_strides(tensor_a.shape, tensor_a.stride, out_shape);
  Shape b_strides = broadcast_strides(tensor_b.shape, tensor_b.stride, out_shape);

  Tensor<T, D> out{out_shape};
  auto out_span = out.span();
  auto a_span = tensor_a.span();
  auto b_span = tensor_b.span();

  // Total elements in output
  size_t total = out.size();

  for (size_t out_idx = 0; out_idx < total; ++out_idx) {
    // Convert flat index to N-dimensional indices
    // Then compute a_idx and b_idx using broadcast strides

    size_t a_idx = 0;
    size_t b_idx = 0;
    size_t remainder = out_idx;

    for (size_t dim = 0; dim < out_shape.size(); ++dim) {
      // How many elements in all dims after this one?
      size_t dim_stride = 1;
      for (size_t the_dim = dim + 1; the_dim < out_shape.size(); ++the_dim) {
        dim_stride *= out_shape[the_dim];
      }

      size_t coord = remainder / dim_stride; // index in this dimension
      remainder = remainder % dim_stride;

      a_idx += coord * a_strides[dim];
      b_idx += coord * b_strides[dim];
    }

    out_span[out_idx] = func(a_span[a_idx], b_span[b_idx]);
  }

  return out;
}

template <DType T, Device D>
Tensor<T, D> add(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise(tensor_a, tensor_b, [](T val_a, T val_b) { return val_a + val_b; });
}

template Tensor<bfloat16, CPU> add(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> add(const TensorView<float, CPU>&, const TensorView<float, CPU>&);
template Tensor<int, CPU> add(const TensorView<int, CPU>&, const TensorView<int, CPU>&);

template <DType T, Device D>
Tensor<T, D> mul(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise(tensor_a, tensor_b, [](T val_a, T val_b) { return val_a * val_b; });
}

template Tensor<bfloat16, CPU> mul(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> mul(const TensorView<float, CPU>&, const TensorView<float, CPU>&);

template <DType T, Device D> Tensor<T, D> mul(const TensorView<T, D>& tensor, T scalar) {
  return tensor.template map<T>([scalar](T val) { return scalar * val; });
}

template Tensor<bfloat16, CPU> mul(const TensorView<bfloat16, CPU>& tensor, bfloat16 scalar);
template Tensor<float, CPU> mul(const TensorView<float, CPU>& tensor, float scalar);

template <DType T, Device D> Tensor<T, D> pow(T scalar, const TensorView<T, D>& tensor) {
  return tensor.template map<T>([scalar](T val) { return std::pow(scalar, val); });
}

template Tensor<bfloat16, CPU> pow(bfloat16 scalar, const TensorView<bfloat16, CPU>& tensor);
template Tensor<float, CPU> pow(float scalar, const TensorView<float, CPU>& tensor);

template <DType T, Device D> Tensor<T, D> pow(const TensorView<T, D>& tensor, T scalar) {
  return tensor.template map<T>([scalar](T val) { return std::pow(val, scalar); });
}

template Tensor<bfloat16, CPU> pow(const TensorView<bfloat16, CPU>& tensor, bfloat16 scalar);
template Tensor<float, CPU> pow(const TensorView<float, CPU>& tensor, float scalar);

// matmul

template <DType T, Device D>
Tensor<T, D> matmul(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  // a: [..., M, K]
  // b: [K, N] (2D) or [..., K, N] (batched)
  // out: [..., M, N]
  //
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

template Tensor<bfloat16, CPU> matmul(const TensorView<bfloat16, CPU>&,
                                      const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> matmul(const TensorView<float, CPU>&, const TensorView<float, CPU>&);

template <DType T, Device D>
Tensor<T, D> cat(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b, int dim) {
  auto shape_a = tensor_a.shape;
  auto shape_b = tensor_b.shape;

  if (dim == -1) {
    dim = shape_a.size() - 1;
  }

  Shape new_shape{shape_a};
  new_shape[dim] += shape_b[dim];

  // flatten the batch dimensions before dim
  size_t outer_iterations = 1;
  for (size_t i = 0; i < dim && i < shape_a.size(); ++i) {
    outer_iterations *= shape_a[i];
  }

  // flatten the dimensions after dim
  size_t chunk_size_a = 1;
  size_t chunk_size_b = 1;
  for (size_t i = dim; i >= dim && i < shape_a.size(); ++i) {
    chunk_size_a *= shape_a[i];
    chunk_size_b *= shape_b[i];
  }

  Tensor<T, D> out{new_shape};

  auto a_span = tensor_a.span();
  auto b_span = tensor_b.span();
  auto out_span = out.span();

  size_t a_offset = 0;
  size_t b_offset = 0;
  size_t out_offset = 0;

  for (size_t i = 0; i < outer_iterations; ++i) {
    std::copy(&a_span[a_offset], &a_span[a_offset + chunk_size_a], &out_span[out_offset]);
    out_offset += chunk_size_a;

    std::copy(&b_span[b_offset], &b_span[b_offset + chunk_size_b], &out_span[out_offset]);
    out_offset += chunk_size_b;

    a_offset += chunk_size_a;
    b_offset += chunk_size_b;
  }

  return out;
}

template Tensor<bfloat16, CPU> cat(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&, int);
template Tensor<float, CPU> cat(const TensorView<float, CPU>&, const TensorView<float, CPU>&, int);

template <DType T, Device D>
Tensor<T, D> slice(const TensorView<T, D>& view, int dim, size_t start, size_t end) {
  auto shape = view.shape;

  if (dim == -1) {
    dim = shape.size() - 1;
  }

  Shape new_shape{shape};
  new_shape[dim] = end - start;

  // flatten the batch dimensions before dim
  size_t outer_iterations = 1;
  for (size_t i = 0; i < dim && i < shape.size(); ++i) {
    outer_iterations *= shape[i];
  }

  // Product of dimensions AFTER dim (not including dim)
  size_t inner_stride = 1;
  for (size_t i = dim + 1; i < shape.size(); ++i) {
    inner_stride *= shape[i];
  }

  // How many elements to copy per outer iteration
  size_t chunk_to_copy = (end - start) * inner_stride;

  // How far to advance in source per outer iteration (full original size)
  size_t source_stride = shape[dim] * inner_stride;

  Tensor<T, D> out{new_shape};

  auto view_span = view.span();
  auto out_span = out.span();

  size_t source_offset = 0;
  size_t out_offset = 0;

  for (size_t i = 0; i < outer_iterations; ++i) {
    // Skip to 'start' position, then copy the slice
    size_t read_from = source_offset + (start * inner_stride);

    std::copy(&view_span[read_from], &view_span[read_from + chunk_to_copy], &out_span[out_offset]);

    // Advance output by what we copied
    out_offset += chunk_to_copy;

    // Advance source by the FULL original row size
    source_offset += source_stride;
  }

  return out;
}

template Tensor<bfloat16, CPU> slice(const TensorView<bfloat16, CPU>& view, int dim, size_t start,
                                     size_t end);
template Tensor<float, CPU> slice(const TensorView<float, CPU>& view, int dim, size_t start,
                                  size_t end);
} // namespace tensor
