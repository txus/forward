#pragma once

#include <tensor/tensor.hpp>

namespace nn {

struct Softmax {
  template <tensor::DType T, tensor::Device D>
  tensor::Tensor<std::remove_const_t<T>, D> operator()(const tensor::TensorView<T, D>& input,
                                                       int dim) const;
};

} // namespace nn
