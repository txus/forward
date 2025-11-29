#include <nn/mlp.hpp>
#include <tensor/ops.hpp>
#include <utility>

using namespace nn;
using namespace tensor;

template <DType T, Device D> void MLP<T, D>::set_weights(TensorView<T, D> weights) {
  weights_ = std::move(weights);
}

template <DType T, Device D> Tensor<T, D> MLP<T, D>::forward(TensorView<T, D> inputs) const {
  const auto w_shape = weights_.shape;

  const size_t w_input_size = w_shape[0];
  const size_t w_output_size = w_shape[1];

  const size_t batch_size = inputs.shape[0];
  const size_t input_size = inputs.shape[1];

  assert(w_input_size == input_size);

  return matmul<T, D>(inputs, weights_);
}

template class nn::MLP<bfloat16, CPU>;
