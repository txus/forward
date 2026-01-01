#include "rms_norm.cuh"
#include "utils.cuh"
#include <tensor/device_type.hpp>

namespace nn::kernels {

using namespace tensor;
using namespace tensor::dtype;

const int blockThreads = 512;

template <typename T>
inline __device__ void calculate_rms(const T* inputs, size_t seq_start, T eps, size_t hidden_dim, T shmem[]) {
  auto tid = threadIdx.x;
  // reduce with a grid stride loop to handle reduce_size > blockThreads
  float thread_sum = 0.0;
  for (size_t channel_idx = tid; channel_idx < hidden_dim; channel_idx += blockDim.x) {
    T val = inputs[seq_start + channel_idx];
    thread_sum += powf(val, 2);
  }

  // load partial sums onto shmem
  shmem[tid] = thread_sum;
  __syncthreads();

  // reduce in shared memory
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) { // NOLINT
    if (tid < stride) { shmem[tid] += shmem[tid + stride]; }
    __syncthreads();
  }

  // warp shuffle for the final warp-level reduction
  if (tid < 32) {
    float val = shmem[tid] + shmem[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (tid == 0) {
      shmem[0] = T(1.0f / sqrtf(val / hidden_dim + float(eps)));
    }
  }
  __syncthreads();
}



template <typename T>
__global__ void rms_norm_kernel(T* out, const T* inputs, const T* weights, T eps, size_t hidden_dim) {
  __shared__ T shmem[blockThreads];

  auto seq_len = gridDim.y;

  auto batch_idx = blockIdx.x;
  auto pos_idx = blockIdx.y;

  auto batch_start = batch_idx * seq_len;
  auto seq_start = batch_start + (pos_idx * hidden_dim);

  calculate_rms(inputs, seq_start, eps, hidden_dim, shmem);
  auto rms = shmem[0];

  for (size_t channel_idx = threadIdx.x; channel_idx < hidden_dim; channel_idx += blockDim.x) {
    if (channel_idx < hidden_dim) {
      out[seq_start + channel_idx] = inputs[seq_start + channel_idx] * rms * weights[channel_idx];
    }
  }
}

template <typename T>
Tensor<T, CUDA> rms_norm(const TensorView<T, CUDA>& input, const TensorView<const T, CUDA>& weights, T eps) {
  assert(input.is_contiguous() && "input should be contiguous");
  assert(weights.is_contiguous() && "weights should be contiguous");

  const auto w_shape = weights.shape;

  const unsigned int batch_size = input.shape[0];
  const unsigned int seq_len = input.shape[1];
  const unsigned int hidden_dim = input.shape[2];

  size_t n_elements = input.data_size;
  TensorStorage<T, CUDA> storage(n_elements);

  Tensor<T, CUDA> out{input.shape, std::move(storage)};

  size_t block_size = blockThreads;
  dim3 grid_size{batch_size, seq_len};

  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<Cuda<T>*>(input.data); // NOLINT
  auto* w_d = reinterpret_cast<const Cuda<T>*>(weights.data); // NOLINT
  Cuda<T> eps_d = to_device_type(eps, CUDA{});

  rms_norm_kernel<Cuda<T>><<<grid_size, block_size>>>(out_d, in_d, w_d, eps_d, hidden_dim);

  return out;
}

template Tensor<bfloat16, CUDA> rms_norm(const TensorView<bfloat16, CUDA>& input, const TensorView<const bfloat16, CUDA>& weights, bfloat16 eps);

}
