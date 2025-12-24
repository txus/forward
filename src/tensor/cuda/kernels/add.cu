#include "add.cuh"
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

template<typename DeviceT>
__global__ void add_kernel(DeviceT* out, DeviceT* tensor_a, DeviceT* tensor_b, size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < n) {
    out[idx] = tensor_a[idx] + tensor_b[idx];
  }
}

// Explicit instantiations
template __global__ void add_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>*, Cuda<float>*, size_t);
template __global__ void add_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>*, Cuda<int>*, size_t);
template __global__ void add_kernel<Cuda<bfloat16>>(Cuda<bfloat16>*, Cuda<bfloat16>*, Cuda<bfloat16>*, size_t);

} // namespace tensor::kernels
