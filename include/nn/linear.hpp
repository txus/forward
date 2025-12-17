#pragma once

#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace nn {
template <tensor::DType T, tensor::Device D> class Linear {
private:
  tensor::TensorView<const T, D> weights_;

public:
  Linear() = default;
  ~Linear() = default;

  void load_weights(const tensor::Loader<T, D>& loader, std::string_view name,
                    bool transpose = false);
  void load_weights(tensor::TensorView<const T, D> weights, bool transpose);

  tensor::Tensor<std::remove_const_t<T>, D> forward(const tensor::TensorView<T, D>& inputs);
};
} // namespace nn
