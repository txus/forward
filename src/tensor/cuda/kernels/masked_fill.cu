#include "masked_fill.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cuda_bf16.hpp>

namespace tensor::kernels {

using namespace dtype;

__global__ void masked_fill_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* input, Cuda<int>* mask, Cuda<bfloat16> masked_value, size_t inner_size) {
  // there are as many blocks as operations to do, which one are we doing?
  size_t operation_idx = blockIdx.x;

  size_t tid = threadIdx.x;

  auto base = operation_idx * inner_size;

  for (size_t element = tid; element < inner_size; element += blockDim.x) {
    if (element < inner_size) {
      Cuda<bfloat16> value = input[base + element];
      Cuda<int> gate = mask[element];

      if (gate == 1) {
        out[base + element] = value;
      } else {
        out[base + element] = masked_value;
      }
    }
  }
}

Tensor<bfloat16, CUDA> masked_fill_bfloat16(const TensorView<bfloat16, CUDA>& input, const TensorView<int, CUDA>& mask, bfloat16 masked_value) {
  auto mask_dims = mask.shape.size();
  auto dims_to_skip = input.shape.size() - mask_dims;

  assert(input.is_contiguous() && mask.is_contiguous() && "input and mask should both be contiguous");

  size_t inner_size = 1;
  for (size_t idx = dims_to_skip; idx < input.shape.size(); ++idx) {
    assert(input.shape[idx] == mask.shape[idx - dims_to_skip] && "the last dimensions of input and mask need to match for broadcasting to work");
    inner_size *= input.shape[idx];
  }

  size_t outer_size = 1;
  for (size_t idx = 0; idx < dims_to_skip; ++idx) {
    outer_size *= input.shape[idx];
  }

  size_t n_elements = input.data_size;
  TensorStorage<bfloat16, CUDA> storage(n_elements);
  Tensor<bfloat16, CUDA> out{input.shape, std::move(storage)};

  size_t block_size = cuda::get_block_size(inner_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<bfloat16>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<Cuda<bfloat16>*>(input.data); // NOLINT
  auto* mask_d = reinterpret_cast<int*>(mask.data); // NOLINT
  Cuda<bfloat16> mask_value_d = to_device_type(masked_value, CUDA{});

  masked_fill_bfloat16_kernel<<<outer_size, block_size>>>(out_d, in_d, mask_d, mask_value_d, inner_size);

  return out;
}

} // namespace tensor::kernels
