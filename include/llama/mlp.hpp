#pragma once

#include <llama/config.hpp>
#include <nn/act.hpp>
#include <nn/linear.hpp>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

namespace llama {
template <tensor::DType T, tensor::Device D> class MLP {
private:
  nn::Linear<T, D> up_proj;
  nn::Linear<T, D> gate_proj;
  nn::Linear<T, D> down_proj;
  nn::Activation act_fn;

public:
  explicit MLP(const ModelConfig& config);
  ~MLP() = default;

  void load_weights(const tensor::Loader<T, D>& loader, size_t layer_idx);

  tensor::Tensor<std::remove_const_t<T>, D> forward(tensor::TensorView<T, D> inputs);
};
} // namespace llama
