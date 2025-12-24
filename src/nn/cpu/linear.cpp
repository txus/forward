#include <nn/linear.hpp>
#include <tensor/loader.hpp>
#include <tensor/ops.hpp>

using namespace nn;
using namespace tensor;

template <DType T, Device D>
void Linear<T, D>::load_weights(const tensor::Loader<T, D>& loader, std::string_view name) {
  weights_ = loader.load(name);
  weights_t_ = weights_.view();
  weights_t_.transpose();
}

template <DType T, Device D>
Tensor<T, D> Linear<T, D>::forward(const TensorView<T, D>& inputs) {
  return matmul(inputs, weights_t_);
}

template class nn::Linear<bfloat16, CPU>;
