#include <nn/linear.hpp>
#include <tensor/ops.hpp>
#include <utility>

using namespace nn;
using namespace tensor;

template <DType T, Device D> void Linear<T, D>::set_weights(TensorView<T, D> weights) {
  weights_ = std::move(weights);
  weights_.transpose();
}

template <DType T, Device D> Tensor<T, D> Linear<T, D>::forward(TensorView<T, D> inputs) const {
  return matmul<T, D>(std::move(inputs), weights_);
}

template class nn::Linear<bfloat16, CPU>;
