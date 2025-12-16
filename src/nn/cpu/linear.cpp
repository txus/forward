#include <nn/linear.hpp>
#include <tensor/ops.hpp>
#include <utility>

using namespace nn;
using namespace tensor;

template <DType T, Device D>
void Linear<T, D>::set_weights(TensorView<T, D> weights, bool transpose) {
  weights_ = std::move(weights);
  if (transpose) {
    weights_.transpose();
  }
}

template <DType T, Device D> Tensor<T, D> Linear<T, D>::forward(TensorView<T, D> inputs) const {
  // fmt::println("{} @ {}", inputs.shape, weights_.shape);
  // fmt::println("{} @ {}", inputs, weights_);
  return matmul<T, D>(std::move(inputs), weights_);
}

template class nn::Linear<bfloat16, CPU>;
