#pragma once

#include <llama/config.hpp>
#include <tensor/tensor.hpp>

namespace llama {

template <typename T, typename D> class RoPE {
private:
  std::tuple<tensor::Tensor<float, D>, tensor::Tensor<float, D>> cos_sin; // float32

public:
  explicit RoPE(const llama::ModelConfig& config);
  ~RoPE() = default;
  RoPE(RoPE&&) noexcept = default;
  RoPE& operator=(RoPE&&) noexcept = default;
  RoPE(const RoPE&) = delete;
  RoPE& operator=(const RoPE&) = delete;

  tensor::TensorView<const float, D> cos() const;
  tensor::TensorView<const float, D> sin() const;

  tensor::Tensor<std::remove_const_t<T>, D> forward(tensor::TensorView<T, D> inputs,
                                                    size_t position_offset = 0) const;
};
} // namespace llama
