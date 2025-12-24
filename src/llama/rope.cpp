#include <fmt/core.h>

#include <cmath>
#include <llama/rope.hpp>
#include <tensor/ops.hpp>
#include <tuple>

using namespace llama;
using namespace tensor;

template <typename D>
inline std::tuple<Tensor<float, D>, Tensor<float, D>>
precompute_rope_values(size_t head_dim, float theta_base, size_t context_length) {
  assert(head_dim % 2 == 0);

  // compute the inverse frequencies
  Tensor<int, D> range = arange<int, D>(0, head_dim, 2);
  auto range_float = range.view().template to<float>();

  auto scaled = div(range_float.view(), float(head_dim));

  auto powd = pow(theta_base, scaled.view());
  auto inv_freq_ = pow(powd.view(), float(-1.0));

  // Apply LLaMA 3 RoPE scaling
  // Parameters from config
  float factor = 32.0;
  float low_freq_factor = 1.0;
  float high_freq_factor = 4.0;
  float old_context_len = 8192.0;

  float low_freq_wavelen = old_context_len / low_freq_factor;   // 8192
  float high_freq_wavelen = old_context_len / high_freq_factor; // 2048

  // For each frequency, compute wavelength and apply scaling
  for (size_t i = 0; i < inv_freq_.size(); ++i) {
    float inv_f = inv_freq_.span()[i];
    float wavelen = 2.0 * M_PI / inv_f;

    if (wavelen < high_freq_wavelen) {
      // High frequency: no scaling
      // inv_freq stays the same
    } else if (wavelen > low_freq_wavelen) {
      // Low frequency: scale down by factor
      inv_freq_.span()[i] = inv_f / factor;
    } else {
      // Medium frequency: smooth interpolation
      float smooth =
          (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
      float scaled_inv_freq = (1.0 - smooth) * (inv_f / factor) + smooth * inv_f;
      inv_freq_.span()[i] = scaled_inv_freq;
    }
  }

  Tensor<float, D> positions = arange<float, D>(float(0.0), float(context_length), float(1.0));

  Tensor<float, D> lhs = positions.view().reshape({positions.shape()[0], 1});
  Tensor<float, D> rhs = inv_freq_.view().reshape({1, inv_freq_.shape()[0]});

  Tensor<float, D> angles = matmul(lhs.view(), rhs.view()); // context_length, head_dim // 2

  angles = cat(angles.view(), angles.view(), 1); // context length, head_Dim

  auto sin = angles.view().sin();
  auto cos = angles.view().cos();

  return std::make_tuple(std::move(cos), std::move(sin));
}

template <typename T, typename D>
RoPE<T, D>::RoPE(const ModelConfig& config)
    : cos_sin(precompute_rope_values<D>(config.head_dim, config.rope_theta,
                                        config.max_position_embeddings)){};

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> RoPE<T, D>::forward(TensorView<T, D> inputs,
                                                      size_t position_offset) const {
  const auto& cos = std::get<0>(cos_sin);
  const auto& sin = std::get<1>(cos_sin);

  Shape shape = inputs.shape;
  size_t batch_size = shape[0];
  size_t num_heads = shape[1];
  size_t seq_len = shape[2];
  size_t head_dim = shape[3];

  assert(head_dim % 2 == 0);

  // Copy inputs to a tensor (stay in bfloat16)
  Tensor<T, D> inputs_t = inputs.copy();

  // Slice and convert cos/sin to bfloat16
  auto adj_cos_ = slice(cos.view(), 0, position_offset, position_offset + seq_len);
  auto adj_cos_bf16 = adj_cos_.view().template to<T>();
  auto adj_cos = adj_cos_bf16.view().reshape({1, 1, seq_len, head_dim});

  auto adj_sin_ = slice(sin.view(), 0, position_offset, position_offset + seq_len);
  auto adj_sin_bf16 = adj_sin_.view().template to<T>();
  auto adj_sin = adj_sin_bf16.view().reshape({1, 1, seq_len, head_dim});

  // Split input into halves
  auto first_half = slice(inputs_t.view(), -1, 0, head_dim / 2);
  auto second_half = slice(inputs_t.view(), -1, head_dim / 2, head_dim);

  // Negate second half
  auto second_half_neg = mul(second_half.view(), T(-1.0));

  // Concatenate as [-x2, x1]
  auto rotated = cat(second_half_neg.view(), first_half.view(), -1);

  // Apply rotation: inputs * cos + rotated * sin
  auto input_cos = mul(inputs_t.view(), adj_cos.view());
  auto rotated_sin = mul(rotated.view(), adj_sin.view());

  auto out = add(input_cos.view(), rotated_sin.view());

  return out;
}

template <typename T, typename D> TensorView<const float, D> RoPE<T, D>::cos() const {
  return std::get<0>(cos_sin).view();
}

template <typename T, typename D> TensorView<const float, D> RoPE<T, D>::sin() const {
  return std::get<1>(cos_sin).view();
}

template class llama::RoPE<bfloat16, CPU>;
