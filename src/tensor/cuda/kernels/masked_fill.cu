#include "masked_fill.cuh"
#include "broadcast.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cuda_bf16.hpp>

namespace tensor::kernels {

using namespace dtype;

// Kernel that handles full broadcasting between input and mask
__global__ void masked_fill_broadcast_kernel(Cuda<bfloat16>* out, const Cuda<bfloat16>* input,
                                              const Cuda<int>* mask, Cuda<bfloat16> masked_value,
                                              const size_t* out_shape, const size_t* mask_strides,
                                              size_t ndim, size_t total_elements) {
  size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_idx < total_elements) {
    size_t mask_idx = broadcast_index(out_idx, out_shape, mask_strides, ndim);

    Cuda<bfloat16> value = input[out_idx];
    Cuda<int> gate = mask[mask_idx];

    if (gate == 1) {
      out[out_idx] = value;
    } else {
      out[out_idx] = masked_value;
    }
  }
}

Tensor<bfloat16, CUDA> masked_fill_bfloat16(const TensorView<bfloat16, CUDA>& input, const TensorView<int, CUDA>& mask, bfloat16 masked_value) {
  assert(input.is_contiguous() && mask.is_contiguous() && "input and mask should both be contiguous");

  // Compute broadcast configuration (input is "a", mask is "b")
  auto config = compute_broadcast_config(input.shape, mask.shape);

  // For masked_fill, output shape should match input shape
  assert(config.out_shape == input.shape && "masked_fill requires mask to broadcast to input shape");

  size_t total_elements = input.data_size;
  TensorStorage<bfloat16, CUDA> storage(total_elements);
  Tensor<bfloat16, CUDA> out{input.shape, std::move(storage)};

  // Allocate and copy broadcast params to device (only need out_shape and b_strides for mask)
  DeviceBroadcastParams params(config);

  size_t block_size = cuda::get_block_size(total_elements);
  size_t grid_size = (total_elements + block_size - 1) / block_size;

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<bfloat16>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<const Cuda<bfloat16>*>(input.data); // NOLINT
  auto* mask_d = reinterpret_cast<const Cuda<int>*>(mask.data); // NOLINT
  Cuda<bfloat16> mask_value_d = to_device_type(masked_value, CUDA{});

  masked_fill_broadcast_kernel<<<grid_size, block_size>>>(out_d, in_d, mask_d, mask_value_d,
                                                           params.d_out_shape, params.d_b_strides,
                                                           config.ndim, total_elements);

  return out;
}

} // namespace tensor::kernels
