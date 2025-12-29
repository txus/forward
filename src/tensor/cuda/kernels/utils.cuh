#pragma once

// Include libstdc++ headers that use __noinline__ BEFORE cuda_runtime.h
// to avoid macro conflict. See: https://github.com/llvm/llvm-project/issues/57544
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <cstddef>

namespace tensor::cuda {

// CUDA error checking - use the macro version to get correct __FILE__ and __LINE__
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    do {                                                                       \
      cudaError_t err = call;                                                  \
      if (err != cudaSuccess) {                                                \
        throw std::runtime_error(fmt::format("CUDA error at {}:{}: {}",        \
                                __FILE__, __LINE__, cudaGetErrorString(err))); \
      }                                                                        \
    } while (0);                                                               \
  }
// Get optimal block size for 1D kernels
inline size_t get_block_size(size_t problem_size, size_t max_block_size = 512) {
  return std::min(max_block_size, std::bit_ceil(problem_size));
}

// Calculate grid size for 1D kernels
inline size_t get_grid_size(size_t n, size_t block_size) {
  return (n + block_size - 1) / block_size;
}


} // namespace tensor
