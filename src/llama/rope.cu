#include <cuda_runtime.h>

#include <llama/rope.hpp>
#include <tensor/device_type.hpp>

using namespace tensor;

inline size_t get_block_size(size_t problem_size, size_t max_block_size = 512) {
  return std::min(max_block_size, std::bit_ceil(problem_size));
}

// Calculate grid size for 1D kernels
inline size_t get_grid_size(size_t n, size_t block_size) {
  return (n + block_size - 1) / block_size;
}

__global__ void apply_rope_scaling_kernel(float* inv_freq, const float factor, float low_freq_factor, float high_freq_factor, float old_context_len, size_t n_elements) {
  size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  float low_freq_wavelen = old_context_len / low_freq_factor;
  float high_freq_wavelen = old_context_len / high_freq_factor;

  if (idx < n_elements) {
    float inv_f = inv_freq[idx];
    float wavelen = M_PI * 2.0 / inv_f;

    if (wavelen < high_freq_wavelen) {
      // high frequency: no scaling
    } else if (wavelen > low_freq_wavelen) {
      // low frequency: scale down by factor
      inv_freq[idx] = inv_f / factor;
    } else {
      // medium frequency: smooth interpolation
      float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
      float scaled_inv_freq = ((1.0 - smooth) * (inv_f / factor)) + (smooth * inv_f);
      inv_freq[idx] = scaled_inv_freq;
    }
  }
}

namespace llama {

template <>
void apply_rope_scaling_<CUDA>(Tensor<float, CUDA>& inv_freq, float factor, float low_freq_factor,
                               float high_freq_factor, float old_context_len) {
  size_t n_elements = inv_freq.size();

  size_t block_size = get_block_size(n_elements);
  size_t grid_size = get_grid_size(n_elements, block_size);

  float* out_d = inv_freq.data();

  apply_rope_scaling_kernel<<<grid_size, block_size>>>(out_d, factor, low_freq_factor, high_freq_factor, old_context_len, n_elements);
}

} // namespace llama
