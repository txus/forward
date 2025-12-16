#include <algorithm>
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

template <DType In1T, DType In2T, DType OutT, Device D, typename Func>
Tensor<OutT, D> element_wise(const TensorView<In1T, D>& tensor_a,
                             const TensorView<In2T, D>& tensor_b, Func func) {
  Shape out_shape = broadcast_shape(tensor_a.shape, tensor_b.shape);
  Shape a_strides = broadcast_strides(tensor_a.shape, tensor_a.stride, out_shape);
  Shape b_strides = broadcast_strides(tensor_b.shape, tensor_b.stride, out_shape);

  Tensor<OutT, D> out{out_shape};
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
  return element_wise<T, T, T, D>(tensor_a, tensor_b,
                                  [](T val_a, T val_b) { return val_a + val_b; });
}

template Tensor<bfloat16, CPU> add(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> add(const TensorView<float, CPU>&, const TensorView<float, CPU>&);
template Tensor<int, CPU> add(const TensorView<int, CPU>&, const TensorView<int, CPU>&);

template <DType T, Device D>
Tensor<T, D> sub(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise<T, T, T, D>(tensor_a, tensor_b,
                                  [](T val_a, T val_b) { return val_a - val_b; });
}

template Tensor<bfloat16, CPU> sub(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> sub(const TensorView<float, CPU>&, const TensorView<float, CPU>&);
template Tensor<int, CPU> sub(const TensorView<int, CPU>&, const TensorView<int, CPU>&);

template <DType T, Device D> Tensor<T, D> sub(const TensorView<T, D>& tensor, T scalar) {
  return tensor.template map<T>([scalar](T val) { return val - scalar; });
}

template Tensor<bfloat16, CPU> sub(const TensorView<bfloat16, CPU>& tensor, bfloat16 scalar);
template Tensor<float, CPU> sub(const TensorView<float, CPU>& tensor, float scalar);

template <DType T, Device D>
Tensor<T, D> mul(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise<T, T, T, D>(tensor_a, tensor_b,
                                  [](T val_a, T val_b) { return val_a * val_b; });
}

template Tensor<bfloat16, CPU> mul(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> mul(const TensorView<float, CPU>&, const TensorView<float, CPU>&);

template <DType T, Device D> Tensor<T, D> mul(const TensorView<T, D>& tensor, T scalar) {
  return tensor.template map<T>([scalar](T val) { return scalar * val; });
}

template Tensor<bfloat16, CPU> mul(const TensorView<bfloat16, CPU>& tensor, bfloat16 scalar);
template Tensor<float, CPU> mul(const TensorView<float, CPU>& tensor, float scalar);

template <DType T, Device D>
Tensor<T, D> div(const TensorView<T, D>& tensor_a, const TensorView<T, D>& tensor_b) {
  return element_wise<T, T, T, D>(tensor_a, tensor_b,
                                  [](T val_a, T val_b) { return val_a / val_b; });
}

template Tensor<bfloat16, CPU> div(const TensorView<bfloat16, CPU>&,
                                   const TensorView<bfloat16, CPU>&);
template Tensor<float, CPU> div(const TensorView<float, CPU>&, const TensorView<float, CPU>&);

template <DType T, Device D> Tensor<T, D> div(const TensorView<T, D>& tensor, T scalar) {
  return tensor.template map<T>([scalar](T val) { return scalar / val; });
}

template Tensor<bfloat16, CPU> div(const TensorView<bfloat16, CPU>& tensor, bfloat16 scalar);
template Tensor<float, CPU> div(const TensorView<float, CPU>& tensor, float scalar);

template <DType T, Device D>
Tensor<T, D> masked_fill(const TensorView<T, D>& tensor_a, const TensorView<int, D>& mask,
                         T masked_value) {
  return element_wise<T, int, T, D>(tensor_a, mask, [masked_value](T val_a, int mask_val) {
    if (mask_val == 1) {
      return val_a;
    }
    return masked_value;
  });
}
template Tensor<bfloat16, CPU> masked_fill(const TensorView<bfloat16, CPU>& tensor_a,
                                           const TensorView<int, CPU>& mask, bfloat16 masked_value);

template <DType T, Device D> Tensor<T, D> tril(const TensorView<T, D>& tensor, bool diagonal) {
  assert(tensor.shape.size() == 2);
  assert(tensor.is_contiguous());

  // M x N
  size_t last_idx = 0;
  if (diagonal) {
    last_idx = 1;
  }

  auto m_dim = tensor.shape[0];
  auto n_dim = tensor.shape[1];

  auto in_span = tensor.span();

  Tensor<T, D> out{{m_dim, n_dim}};
  out.fill_(0.0);
  auto out_span = out.span();

  for (size_t row = 0; row < m_dim; ++row) {
    auto cols_to_copy = last_idx + 1;
    auto row_offset = row * n_dim;
    std::copy(&in_span[row_offset], &in_span[row_offset + cols_to_copy], &out_span[row_offset]);
    ++last_idx;
  }

  return out;
}
template Tensor<bfloat16, CPU> tril(const TensorView<bfloat16, CPU>& tensor, bool diagonal);
template Tensor<float, CPU> tril(const TensorView<float, CPU>& tensor, bool diagonal);
template Tensor<int, CPU> tril(const TensorView<int, CPU>& tensor, bool diagonal);

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

  if (dim < 0) {
    dim = shape.size() + dim;
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

template <DType T, Device D, typename ReduceFunc, typename InitFunc>
Tensor<T, D> reduce(const TensorView<T, D>& input, int dim, bool keepdim, // NOLINT
                    InitFunc init_fn,                                     // NOLINT
                    ReduceFunc reduce_fn) {                               // NOLINT
  auto shape = input.shape;

  if (dim < 0) {
    dim = shape.size() + dim;
  }

  assert(dim >= 0 && static_cast<size_t>(dim) < shape.size());

  // Output shape
  Shape out_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      if (keepdim) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(shape[i]);
    }
  }

  if (out_shape.empty()) {
    out_shape.push_back(1);
  }

  Tensor<T, D> out{out_shape};
  auto out_span = out.span();

  for (size_t i = 0; i < out.size(); ++i) {
    out_span[i] = init_fn();
  }

  auto in_span = input.span();
  auto in_strides = input.stride;

  size_t total_in = input.data.size();

  for (size_t in_idx = 0; in_idx < total_in; ++in_idx) {
    std::vector<size_t> indices = linear_to_multidim(in_idx, shape);

    std::vector<size_t> out_indices;
    for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
      if (dim_ == static_cast<size_t>(dim)) {
        if (keepdim) {
          out_indices.push_back(0);
        }
      } else {
        out_indices.push_back(indices[dim_]);
      }
    }

    size_t out_idx = 0;
    size_t stride = 1;
    for (int dim_ = out_shape.size() - 1; dim_ >= 0; --dim_) { // NOLINT
      out_idx += out_indices[dim_] * stride;
      stride *= out_shape[dim_];
    }

    size_t in_offset = 0;
    for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
      in_offset += indices[dim_] * in_strides[dim_];
    }

    // Apply reduction
    out_span[out_idx] = reduce_fn(out_span[out_idx], in_span[in_offset]);
  }

  return out;
}

template <DType T, Device D, typename CompareFunc>
Tensor<int, D> reduce_with_index(const TensorView<T, D>& input, int dim, bool keepdim, // NOLINT
                                 CompareFunc compare_fn) {
  auto shape = input.shape;

  if (dim < 0) {
    dim = shape.size() + dim;
  }

  assert(dim >= 0 && static_cast<size_t>(dim) < shape.size());

  // Output shape
  Shape out_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      if (keepdim) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(shape[i]);
    }
  }

  if (out_shape.empty()) {
    out_shape.push_back(1);
  }

  Tensor<int, D> out{out_shape}; // Output is indices (size_t)
  auto out_span = out.span();

  // Track best value for each output position
  std::vector<T> best_values(out.size());

  // Initialize with worst possible values and index 0
  for (size_t i = 0; i < out.size(); ++i) {
    best_values[i] = -std::numeric_limits<T>::infinity();
    out_span[i] = 0;
  }

  auto in_span = input.span();
  auto in_strides = input.stride;
  size_t total_in = input.data.size();

  for (size_t in_idx = 0; in_idx < total_in; ++in_idx) {
    std::vector<size_t> indices = linear_to_multidim(in_idx, shape);

    // Get the index along the reduction dimension
    size_t reduction_idx = indices[dim];

    // Map to output index
    std::vector<size_t> out_indices;
    for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
      if (dim_ == static_cast<size_t>(dim)) {
        if (keepdim) {
          out_indices.push_back(0);
        }
      } else {
        out_indices.push_back(indices[dim_]);
      }
    }

    // Convert to linear output index
    size_t out_idx = 0;
    size_t stride = 1;
    for (int dim_ = out_shape.size() - 1; dim_ >= 0; --dim_) { // NOLINT
      out_idx += out_indices[dim_] * stride;
      stride *= out_shape[dim_];
    }

    // Compute input offset
    size_t in_offset = 0;
    for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
      in_offset += indices[dim_] * in_strides[dim_];
    }

    T value = in_span[in_offset];

    // Update if this is better
    if (compare_fn(value, best_values[out_idx])) {
      best_values[out_idx] = value;
      out_span[out_idx] = reduction_idx; // Store the INDEX
    }
  }

  return out;
}

template <DType T, Device D>
Tensor<T, D> sum(const TensorView<T, D>& input, int dim, bool keepdim) {
  return reduce(
      input, dim, keepdim, []() { return T(0.0); },
      [](T val_a, T val_b) { return T(val_a + val_b); });
}

template Tensor<bfloat16, CPU> sum(const TensorView<bfloat16, CPU>& input, int dim, bool keepdim);
template Tensor<float, CPU> sum(const TensorView<float, CPU>& input, int dim, bool keepdim);
template Tensor<int, CPU> sum(const TensorView<int, CPU>& input, int dim, bool keepdim);

template <DType T, Device D>
Tensor<T, D> max(const TensorView<T, D>& input, int dim, bool keepdim) {
  return reduce(
      input, dim, keepdim, []() { return T(-std::numeric_limits<T>::infinity()); },
      [](T val_a, T val_b) {
        if (val_a >= val_b) {
          return val_a;
        }
        return val_b;
      });
}

template Tensor<bfloat16, CPU> max(const TensorView<bfloat16, CPU>& input, int dim, bool keepdim);
template Tensor<float, CPU> max(const TensorView<float, CPU>& input, int dim, bool keepdim);
template Tensor<int, CPU> max(const TensorView<int, CPU>& input, int dim, bool keepdim);

template <DType T, Device D>
Tensor<int, D> argmax(const TensorView<T, D>& input, int dim, bool keepdim) {
  return reduce_with_index<T, D>(
      input, dim, keepdim,
      [](T val_a, T val_b) { return val_a > val_b; } // Compare: is a better than b?
  );
}

template Tensor<int, CPU> argmax(const TensorView<bfloat16, CPU>& input, int dim, bool keepdim);
template Tensor<int, CPU> argmax(const TensorView<float, CPU>& input, int dim, bool keepdim);
template Tensor<int, CPU> argmax(const TensorView<int, CPU>& input, int dim, bool keepdim);

} // namespace tensor
