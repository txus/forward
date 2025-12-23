#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fmt/core.h>

#include <tensor/ops.hpp>

namespace tensor {

// TODO: Implement CUDA kernels and operations here
//
// Suggested implementation order:
// 1. Element-wise ops (add, mul, etc.) - easiest, good for learning CUDA
// 2. Reductions (sum, max, argmax) - introduce shared memory
// 3. Matrix multiplication (matmul) - use cuBLAS for performance
// 4. Memory ops (cat, slice) - practice with memory patterns
// 5. Advanced ops (tril, masked_fill) - combine techniques

// Example skeleton for element-wise addition:
//
// __global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
//   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     out[idx] = a[idx] + b[idx];
//   }
// }
//
// template <DType T, Device D>
// Tensor<std::remove_const_t<T>, D> add(const TensorView<T, D>& tensor_a,
//                                       const TensorView<T, D>& tensor_b) {
//   // Implementation here
// }

} // namespace tensor
