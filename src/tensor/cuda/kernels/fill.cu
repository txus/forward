#include "fill.cuh"
#include <cstddef>

namespace tensor::kernels {

using namespace dtype;

template<typename DeviceT>
__global__ void fill_kernel(DeviceT* out, DeviceT value, size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    out[idx] = value;
  }
}

template __global__ void fill_kernel<Cuda<float>>(Cuda<float>*, Cuda<float>, size_t);
template __global__ void fill_kernel<Cuda<int>>(Cuda<int>*, Cuda<int>, size_t);
template __global__ void fill_kernel<Cuda<bfloat16>>(Cuda<bfloat16>*, Cuda<bfloat16>, size_t);

} // namespace tensor::kernels
