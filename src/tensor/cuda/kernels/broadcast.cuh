#pragma once

#include <tensor/tensor.hpp>
#include <cassert>
#include <cstddef>

namespace tensor::kernels {

// Maximum dimensions supported for broadcasting
constexpr size_t MAX_BROADCAST_DIMS = 8;

// Host-side struct to hold broadcast configuration
struct BroadcastConfig {
  Shape out_shape;
  Shape a_strides;
  Shape b_strides;
  size_t ndim;
  size_t total_elements;
};

// Compute broadcast configuration for two tensors
// Returns the output shape and broadcast strides for each input
inline BroadcastConfig compute_broadcast_config(const Shape& shape_a, const Shape& shape_b) {
  size_t rank_a = shape_a.size();
  size_t rank_b = shape_b.size();
  size_t ndim = std::max(rank_a, rank_b);

  assert(ndim <= MAX_BROADCAST_DIMS && "too many dimensions for broadcasting");

  // Compute output shape
  Shape out_shape(ndim);
  size_t pad_a = ndim - rank_a;
  size_t pad_b = ndim - rank_b;

  for (size_t i = 0; i < ndim; ++i) {
    size_t dim_a = (i < pad_a) ? 1 : shape_a[i - pad_a];
    size_t dim_b = (i < pad_b) ? 1 : shape_b[i - pad_b];
    assert((dim_a == dim_b || dim_a == 1 || dim_b == 1) && "incompatible shapes for broadcasting");
    out_shape[i] = std::max(dim_a, dim_b);
  }

  // Compute actual strides for contiguous tensors
  Shape a_actual_strides(rank_a);
  if (rank_a > 0) {
    a_actual_strides[rank_a - 1] = 1;
    for (int i = static_cast<int>(rank_a) - 2; i >= 0; --i) {
      a_actual_strides[i] = a_actual_strides[i + 1] * shape_a[i + 1];
    }
  }

  Shape b_actual_strides(rank_b);
  if (rank_b > 0) {
    b_actual_strides[rank_b - 1] = 1;
    for (int i = static_cast<int>(rank_b) - 2; i >= 0; --i) {
      b_actual_strides[i] = b_actual_strides[i + 1] * shape_b[i + 1];
    }
  }

  // Compute broadcast strides (0 where broadcasting)
  Shape a_strides(ndim, 0);
  Shape b_strides(ndim, 0);

  for (size_t i = 0; i < ndim; ++i) {
    // Handle tensor a
    if (i < pad_a) {
      a_strides[i] = 0; // Leading dimension not in a - broadcast
    } else {
      size_t a_dim_idx = i - pad_a;
      size_t a_dim = shape_a[a_dim_idx];
      if (a_dim == 1 && out_shape[i] > 1) {
        a_strides[i] = 0; // Broadcasting
      } else {
        a_strides[i] = a_actual_strides[a_dim_idx];
      }
    }

    // Handle tensor b
    if (i < pad_b) {
      b_strides[i] = 0; // Leading dimension not in b - broadcast
    } else {
      size_t b_dim_idx = i - pad_b;
      size_t b_dim = shape_b[b_dim_idx];
      if (b_dim == 1 && out_shape[i] > 1) {
        b_strides[i] = 0; // Broadcasting
      } else {
        b_strides[i] = b_actual_strides[b_dim_idx];
      }
    }
  }

  // Compute total output elements
  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; ++i) {
    total_elements *= out_shape[i];
  }

  return BroadcastConfig{
      .out_shape = std::move(out_shape),
      .a_strides = std::move(a_strides),
      .b_strides = std::move(b_strides),
      .ndim = ndim,
      .total_elements = total_elements};
}

// Device-side helper to compute input index from output index using broadcast strides
// This is a device function that can be called from kernels
__device__ inline size_t broadcast_index(size_t out_idx, const size_t* out_shape,
                                          const size_t* strides, size_t ndim) {
  size_t in_idx = 0;
  size_t remainder = out_idx;

  for (size_t dim = 0; dim < ndim; ++dim) {
    // Compute stride in output for this dimension
    size_t dim_stride = 1;
    for (size_t d = dim + 1; d < ndim; ++d) {
      dim_stride *= out_shape[d];
    }

    size_t coord = remainder / dim_stride;
    remainder = remainder % dim_stride;

    in_idx += coord * strides[dim];
  }

  return in_idx;
}

// Helper struct for device memory management of broadcast params
struct DeviceBroadcastParams {
  size_t* d_out_shape = nullptr;
  size_t* d_a_strides = nullptr;
  size_t* d_b_strides = nullptr;

  DeviceBroadcastParams(const BroadcastConfig& config) {
    cudaMalloc(&d_out_shape, config.ndim * sizeof(size_t));
    cudaMalloc(&d_a_strides, config.ndim * sizeof(size_t));
    cudaMalloc(&d_b_strides, config.ndim * sizeof(size_t));
    cudaMemcpy(d_out_shape, config.out_shape.data(), config.ndim * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, config.a_strides.data(), config.ndim * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, config.b_strides.data(), config.ndim * sizeof(size_t),
               cudaMemcpyHostToDevice);
  }

  ~DeviceBroadcastParams() {
    if (d_out_shape) cudaFree(d_out_shape);
    if (d_a_strides) cudaFree(d_a_strides);
    if (d_b_strides) cudaFree(d_b_strides);
  }

  // Non-copyable
  DeviceBroadcastParams(const DeviceBroadcastParams&) = delete;
  DeviceBroadcastParams& operator=(const DeviceBroadcastParams&) = delete;

  // Movable
  DeviceBroadcastParams(DeviceBroadcastParams&& other) noexcept
      : d_out_shape(other.d_out_shape),
        d_a_strides(other.d_a_strides),
        d_b_strides(other.d_b_strides) {
    other.d_out_shape = nullptr;
    other.d_a_strides = nullptr;
    other.d_b_strides = nullptr;
  }

  DeviceBroadcastParams& operator=(DeviceBroadcastParams&& other) noexcept {
    if (this != &other) {
      if (d_out_shape) cudaFree(d_out_shape);
      if (d_a_strides) cudaFree(d_a_strides);
      if (d_b_strides) cudaFree(d_b_strides);
      d_out_shape = other.d_out_shape;
      d_a_strides = other.d_a_strides;
      d_b_strides = other.d_b_strides;
      other.d_out_shape = nullptr;
      other.d_a_strides = nullptr;
      other.d_b_strides = nullptr;
    }
    return *this;
  }
};

} // namespace tensor::kernels
