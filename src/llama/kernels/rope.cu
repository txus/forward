#include <cuda_runtime.h>

#include <cstddef>
#include <llama/rope.hpp>
#include <tensor/device_type.hpp>
#include <tensor/device.hpp>

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

template<typename T>
__global__ void rope_fused(T* out, T* inputs, const float* cos, const float* sin,
                           size_t head_dim, size_t heads_per_block, size_t position_offset,
                           size_t in_batch_stride, size_t in_head_stride, size_t in_seq_stride) {
  // Shared memory layout: [cos: head_dim floats][sin: head_dim floats]
  extern __shared__ float shared_mem[]; // NOLINT
  float* sh_cos = reinterpret_cast<float*>(shared_mem); // NOLINT
  float* sh_sin = sh_cos + head_dim;

  size_t seq_idx = blockIdx.x;
  size_t local_head_idx = threadIdx.x / head_dim; // which head in this `heads_per_block` group are we?
  size_t channel_idx = threadIdx.x % head_dim;

  // load an entire head of cos and sin into shared memory, to be reused `heads_per_block` times
  if (local_head_idx == 0) {
    sh_cos[channel_idx] = cos[((position_offset + seq_idx) * head_dim) + channel_idx];
    sh_sin[channel_idx] = sin[((position_offset + seq_idx) * head_dim) + channel_idx];
  }
  __syncthreads();

  size_t head_idx = (blockIdx.y * heads_per_block) + local_head_idx;
  size_t num_heads = gridDim.y * heads_per_block;
  size_t batch_idx = blockIdx.z;
  size_t seq_len = gridDim.x;

  // Output is always contiguous: [batch, heads, seq, head_dim]
  size_t out_head_stride = seq_len * head_dim;
  size_t out_batch_stride = num_heads * out_head_stride;
  size_t out_offset = (batch_idx * out_batch_stride) + (head_idx * out_head_stride) + (seq_idx * head_dim) + channel_idx;

  // Input uses provided strides (may be transposed)
  size_t in_offset = (batch_idx * in_batch_stride) + (head_idx * in_head_stride) + (seq_idx * in_seq_stride) + channel_idx;

  size_t half_head = head_dim / 2;

  auto input_val = float(inputs[in_offset]);

  float sin_val = sh_sin[channel_idx];
  float cos_val = sh_cos[channel_idx];

  float rot_val = 0;
  if (channel_idx < half_head) {
    rot_val = -float(inputs[in_offset + half_head]);
  } else {
    rot_val = float(inputs[in_offset - half_head]);
  }

  out[out_offset] = T((rot_val * sin_val) + (input_val * cos_val));
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


template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> rope_forward_fused(const TensorView<T, D> &inputs,
                                                     const TensorView<const float, D> &cos,
                                                     const TensorView<const float, D> &sin,
                                                     size_t position_offset) {
  Shape shape = inputs.shape;
  unsigned int batch_size = shape[0];
  unsigned int num_heads = shape[1];
  unsigned int seq_len = shape[2];
  size_t head_dim = shape[3];

  size_t heads_per_block = std::min(static_cast<size_t>(num_heads), static_cast<size_t>(2));

  assert(head_dim % 2 == 0);
  assert(num_heads % heads_per_block == 0 && "num_heads must be divisible by heads_per_block");

  // Extract input strides for strided kernel (no copy needed!)
  // Shape is [batch, heads, seq, head_dim], strides tell us actual memory layout
  size_t in_batch_stride = inputs.stride[0];
  size_t in_head_stride = inputs.stride[1];
  size_t in_seq_stride = inputs.stride[2];
  // stride[3] is always 1 for head_dim (innermost contiguous dimension)

  auto n_elements = static_cast<size_t>(batch_size * num_heads * seq_len) * head_dim;

  TensorStorage<T, CUDA> storage(n_elements);
  Tensor<T, CUDA> out{shape, std::move(storage)};

  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<Cuda<T>*>(inputs.data); // NOLINT
  auto* cos_d = cos.data;
  auto* sin_d = sin.data;

  dim3 grid_size{seq_len, num_heads / static_cast<unsigned int>(heads_per_block), batch_size};
  size_t block_size = head_dim * heads_per_block;
  size_t shared_mem_size = head_dim * sizeof(Cuda<float>) * 2; // cos + sin

  rope_fused<Cuda<T>><<<grid_size, block_size, shared_mem_size>>>(
      out_d, in_d, cos_d, sin_d, head_dim, heads_per_block, position_offset,
      in_batch_stride, in_head_stride, in_seq_stride);

  return out;
}

// Explicit instantiation for the only currently used type
template Tensor<bfloat16, CUDA> rope_forward_fused<bfloat16, CUDA>(
  const TensorView<bfloat16, CUDA> &inputs,
  const TensorView<const float, CUDA> &cos,
  const TensorView<const float, CUDA> &sin,
  size_t position_offset);

} // namespace llama
