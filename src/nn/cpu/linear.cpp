#include <nn/linear.hpp>
#include <tensor/loader.hpp>
#include <tensor/ops.hpp>

using namespace nn;
using namespace tensor;

template <DType T, Device D>
void Linear<T, D>::load_weights(const tensor::Loader<T, D>& loader, std::string_view name,
                                bool transpose) {
  weights_ = loader.load(name, transpose);
}

template <DType T, Device D>
void Linear<T, D>::load_weights(TensorView<const T, D> weights, bool transpose) {
  weights_ = std::move(weights);
  if (transpose) {
    weights_.transpose(0, 1);
  }
}

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> Linear<T, D>::forward(const TensorView<T, D>& inputs) {
  // fmt::println("{} @ {}", inputs.shape, weights_.shape);
  // fmt::println("{} @ {}", inputs, weights_);

  return matmul(inputs, weights_);
}

template class nn::Linear<bfloat16, CPU>;
