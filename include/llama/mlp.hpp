#pragma once

#include <llama/config.hpp>
#include <nn/act.hpp>
#include <nn/linear.hpp>
#include <tensor/tensor.hpp>
#include <unordered_map>

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

  void load_weights(std::unordered_map<std::string, tensor::Tensor<T, D>>& weight_map,
                    size_t layer_idx);

  tensor::Tensor<T, D> forward(tensor::TensorView<T, D> inputs) const;
};
} // namespace llama
