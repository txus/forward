#include <tensor/dtype.hpp>
#include "fill.cuh"
#include <cstddef>

using namespace tensor::dtype;

template<DType T>
__global__ void fill_kernel(T* out, T value, size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    out[idx] = value;
  }
}

// Explicit instantiations
template __global__ void fill_kernel<float>(float*, float, size_t);
template __global__ void fill_kernel<int>(int*, int, size_t);
template __global__ void fill_kernel<bfloat16>(bfloat16*, bfloat16, size_t);
