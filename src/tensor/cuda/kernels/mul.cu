#include "mul.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cuda_bf16.hpp>

namespace tensor::kernels {

using namespace dtype;

__global__ void mul_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor_a, Cuda<bfloat16>* tensor_b, size_t n) {
  // we load 8 bf16 values at a time = 128 bits
  auto base = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto idx = base * 8;

  if (idx + 7 < n) {
    // load 128 bits
    uint4 a_vec = reinterpret_cast<uint4*>(tensor_a)[base]; // NOLINT
    uint4 b_vec = reinterpret_cast<uint4*>(tensor_b)[base]; // NOLINT

    // reinterpret as pairs of bf16s
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&a_vec); // NOLINT
    __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(&b_vec); // NOLINT

    uint4 out_vec;
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(&out_vec); // NOLINT

    out2[0] = __hmul2(a2[0], b2[0]);
    out2[1] = __hmul2(a2[1], b2[1]);
    out2[2] = __hmul2(a2[2], b2[2]);
    out2[3] = __hmul2(a2[3], b2[3]);

    reinterpret_cast<uint4*>(out)[base] = out_vec; // NOLINT
  }
}

Tensor<bfloat16, CUDA> mul_bfloat16(const TensorView<bfloat16, CUDA>& tensor_a, const TensorView<bfloat16, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");
  assert(tensor_a.shape == tensor_b.shape && "the two tensors should be the same shape");

  size_t n_elements = tensor_a.data_size;
  TensorStorage<bfloat16, CUDA> storage(n_elements);

  Tensor<bfloat16, CUDA> out{tensor_a.shape, std::move(storage)};

  int block_size = 512;
  // each thread handles 8 elements
  int grid_size = cuda::get_grid_size(n_elements / 8, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<bfloat16>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<bfloat16>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<bfloat16>*>(tensor_b.data); // NOLINT

  mul_bfloat16_kernel<<<grid_size, block_size>>>(out_d, a_d, b_d, n_elements);

  return out;
}

__global__ void mul_bfloat16_kernel(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor, Cuda<bfloat16> scalar, size_t n) {
  // we load 8 bf16 values at a time = 128 bits
  auto base = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto idx = base * 8;

  const __nv_bfloat162 twice_scalar{scalar, scalar};

  if (idx + 7 < n) {
    // load 128 bits
    uint4 a_vec = reinterpret_cast<uint4*>(tensor)[base]; // NOLINT

    // reinterpret as pairs of bf16s
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&a_vec); // NOLINT

    uint4 out_vec;
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(&out_vec); // NOLINT

    out2[0] = __hmul2(a2[0], twice_scalar);
    out2[1] = __hmul2(a2[1], twice_scalar);
    out2[2] = __hmul2(a2[2], twice_scalar);
    out2[3] = __hmul2(a2[3], twice_scalar);

    reinterpret_cast<uint4*>(out)[base] = out_vec; // NOLINT
  }
}

Tensor<bfloat16, CUDA> mul_bfloat16(const TensorView<bfloat16, CUDA>& tensor, bfloat16 scalar) {
  assert(tensor.is_contiguous() && "the two tensors should be contiguous");

  size_t n_elements = tensor.data_size;
  TensorStorage<bfloat16, CUDA> storage(n_elements);

  Tensor<bfloat16, CUDA> out{tensor.shape, std::move(storage)};

  int block_size = 512;
  // each thread handles 8 elements
  int grid_size = cuda::get_grid_size(n_elements / 8, block_size);

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<bfloat16>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<Cuda<bfloat16>*>(tensor.data); // NOLINT
  Cuda<float> device_scalar = to_device_type(scalar, CUDA{});

  mul_bfloat16_kernel<<<grid_size, block_size>>>(out_d, a_d, device_scalar, n_elements);

  return out;
}

} // namespace tensor::kernels
