#include "embedding.cuh"
#include "utils.cuh"
#include <tensor/device_type.hpp>

namespace nn::kernels {

using namespace tensor;
using namespace tensor::dtype;

template <typename T>
__global__ void embedding_kernel(T* out, const int* token_ids, const T* weights, size_t hidden_dim) {
  auto seq_len = gridDim.y;

  auto batch_idx = blockIdx.x;
  auto pos_idx = blockIdx.y;

  auto batch_start = batch_idx * seq_len;
  auto seq_start = batch_start + (pos_idx * hidden_dim);

  int token_idx = token_ids[batch_start + pos_idx];

  for (size_t channel_idx = threadIdx.x; channel_idx < hidden_dim; channel_idx += blockDim.x) {
    if (channel_idx < hidden_dim) {
      out[seq_start + channel_idx] = weights[(token_idx * hidden_dim) + channel_idx];
    }
  }
}

template <typename T>
Tensor<T, CUDA> embedding(const TensorView<int, CUDA>& token_ids, const TensorView<const T, CUDA>& weights) {
  assert(token_ids.is_contiguous() && "tokens should be contiguous");
  assert(weights.is_contiguous() && "weights should be contiguous");

  const auto w_shape = weights.shape;

  const unsigned int vocab_size = w_shape[0];
  const unsigned int hidden_dim = w_shape[1];

  const unsigned int batch_size = token_ids.shape[0];
  const unsigned int seq_len = token_ids.shape[1];

  size_t n_elements = batch_size * seq_len * hidden_dim;
  TensorStorage<T, CUDA> storage(n_elements);

  Tensor<T, CUDA> out{{batch_size, seq_len, hidden_dim}, std::move(storage)};

  size_t block_size = cuda::get_block_size(hidden_dim);
  dim3 grid_size{batch_size, seq_len};

  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* tok_d = reinterpret_cast<Cuda<int>*>(token_ids.data); // NOLINT
  auto* w_d = reinterpret_cast<const Cuda<T>*>(weights.data); // NOLINT
                                                        //
  embedding_kernel<Cuda<T>><<<grid_size, block_size>>>(out_d, tok_d, w_d, hidden_dim);

  return out;
}

template Tensor<bfloat16, CUDA> embedding(const TensorView<int, CUDA>& token_ids, const TensorView<const bfloat16, CUDA>& weights);

}
