#include <cmath>
#include <nn/rms_norm.hpp>
#include <tensor/loader.hpp>
#include <util/nvtx.hpp>
#include "kernels/rms_norm.cuh"
#include "kernels/utils.cuh"

using namespace nn;
using namespace tensor;

template <typename T, typename D> RMSNorm<T, D>::RMSNorm(float eps) : eps(eps){};

template <typename T, typename D>
void RMSNorm<T, D>::load_weights(const tensor::Loader<T, D>& loader, std::string_view name) {
  weights_ = loader.load(name);
}

template <typename T, typename D>
void RMSNorm<T, D>::load_weights(tensor::Tensor<const T, D> weights) {
  weights_ = std::move(weights);
}

template <typename T, typename D>
void RMSNorm<T, D>::load_weights(const tensor::Tensor<T, D>& weights) {
  TensorStorage<const T, D> storage(weights.size());
  CUDA_CHECK(cudaMemcpy(storage.mutable_data(), weights.data(),
                        weights.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  weights_ = Tensor<const T, D>{weights.shape(), std::move(storage)};
}

template <typename T, typename D>
Tensor<T, D> RMSNorm<T, D>::forward(const TensorView<T, D>& inputs) const {
  NVTX_RANGE("rms_norm");
  return kernels::rms_norm(inputs, weights_.view(), T(eps));
}

template class nn::RMSNorm<bfloat16, CUDA>;
