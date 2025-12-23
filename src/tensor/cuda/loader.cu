#include <cuda_runtime.h>
#include <fmt/core.h>

#include <tensor/loader.hpp>

namespace tensor {

// TODO: Implement CUDA-specific tensor loading
//
// This should handle:
// 1. Loading tensors directly to GPU memory from safetensors
// 2. Zero-copy loading if possible
// 3. Efficient memory transfers (cudaMemcpy, pinned memory, etc.)
//
// You can likely reuse most of the CPU loader logic and add
// a cudaMemcpy at the end, or use cudaMallocHost for pinned memory
// to speed up host->device transfers.

} // namespace tensor
