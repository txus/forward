#pragma once

#include <llama/config.hpp>
#include <tensor/tensor.hpp>

namespace llama {

using namespace tensor;

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> rope_forward(const TensorView<T, D> &inputs,
                                               const TensorView<const float, D> &cos,
                                               const TensorView<const float, D> &sin,
                                               size_t position_offset);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> rope_forward_fused(const TensorView<T, D> &inputs,
                                                     const TensorView<const float, D> &cos,
                                                     const TensorView<const float, D> &sin,
                                                     size_t position_offset);

template <typename D>
void apply_rope_scaling_(Tensor<float, D>& inv_freq, float factor, float low_freq_factor,
                         float high_freq_factor, float old_context_len);

template <typename T, typename D> class RoPE {
private:
  std::tuple<Tensor<float, D>, Tensor<float, D>> cos_sin; // float32

public:
  explicit RoPE(const llama::ModelConfig& config);
  ~RoPE() = default;
  RoPE(RoPE&&) noexcept = default;
  RoPE& operator=(RoPE&&) noexcept = default;
  RoPE(const RoPE&) = delete;
  RoPE& operator=(const RoPE&) = delete;

  TensorView<const float, D> cos() const;
  TensorView<const float, D> sin() const;

  Tensor<std::remove_const_t<T>, D> forward(const TensorView<T, D> &inputs,
                                            size_t position_offset = 0) const;
};
} // namespace llama
