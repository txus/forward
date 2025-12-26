#include "add.cuh"
#include <cstddef>
#include <cuda_bf16.hpp>

namespace tensor::kernels {

using namespace dtype;

template<typename DeviceT>
__global__ void add_kernel(DeviceT* out, DeviceT* tensor_a, DeviceT* tensor_b, size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < n) {
    out[idx] = tensor_a[idx] + tensor_b[idx];
  }
}

__global__ void add_kernel_bf16(Cuda<bfloat16>* out, Cuda<bfloat16>* tensor_a, Cuda<bfloat16>* tensor_b, size_t n) {
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

    out2[0] = __hadd2(a2[0], b2[0]);
    out2[1] = __hadd2(a2[1], b2[1]);
    out2[2] = __hadd2(a2[2], b2[2]);
    out2[3] = __hadd2(a2[3], b2[3]);

    reinterpret_cast<uint4*>(out)[base] = out_vec; // NOLINT
  }
}

template __global__ void add_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>*, Cuda<float>*, size_t);
template __global__ void add_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>*, Cuda<int>*, size_t);

} // namespace tensor::kernels
