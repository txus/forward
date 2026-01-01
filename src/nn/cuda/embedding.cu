#include <nn/embedding.hpp>
#include <tensor/loader.hpp>
#include <tensor/ops.hpp>
#include <cuda_runtime.h>
#include "kernels/embedding.cuh"
#include "kernels/utils.cuh"

using namespace nn;
using namespace tensor;

template <typename T, typename D>
void Embedding<T, D>::load_weights(const tensor::Loader<T, D>& loader) {
  weights_ = loader.load("model.embed_tokens.weight");
}

template <typename T, typename D>
void Embedding<T, D>::load_weights(tensor::Tensor<const T, D> weights) {
  weights_ = std::move(weights);
}

template <typename T, typename D>
void Embedding<T, D>::load_weights(const tensor::Tensor<T, D>& weights) {
  // For CUDA, allocate const storage and copy data
  TensorStorage<const T, D> storage(weights.size());
  CUDA_CHECK(cudaMemcpy(storage.mutable_data(), weights.data(),
                        weights.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  weights_ = Tensor<const T, D>{weights.shape(), std::move(storage)};
}

template <typename T, typename D>
Tensor<T, D> Embedding<T, D>::forward(const TensorView<int, D>& token_ids) const {
  return kernels::embedding(token_ids, weights_.view());
}

template class nn::Embedding<bfloat16, CUDA>;
