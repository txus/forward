#include <llama/rope.hpp>
#include <tensor/ops.hpp>
#include <tuple>

using namespace llama;
using namespace tensor;

template <Device D>
inline std::tuple<Tensor<float, D>, Tensor<float, D>>
precompute_rope_values(size_t head_dim, float theta_base, size_t context_length) {
  assert(head_dim % 2 == 0);

  // compute the inverse frequencies
  Tensor<int, D> range = arange<int, D>(0, head_dim, 2);
  auto range_float = range.view().template to<float>();
  auto scaled = range_float.view() / float(head_dim);
  auto powd = pow(theta_base, scaled.view());
  auto inv_freq_ = pow(powd.view(), float(-1.0));

  Tensor<float, D> positions = arange<float, D>(float(0.0), float(context_length), float(1.0));

  TensorView<float, D> lhs = positions.view().view_as({positions.shape()[0], 1});
  TensorView<float, D> rhs = inv_freq_.view().view_as({1, inv_freq_.shape()[0]});

  Tensor<float, D> angles = matmul(lhs, rhs);    // context_length, head_dim // 2
                                                 //
  angles = cat(angles.view(), angles.view(), 1); // context length, head_Dim

  auto sin = angles.view().sin();
  auto cos = angles.view().cos();

  auto out = std::tuple(cos, sin);
  return out;
}

template <DType T, Device D>
RoPE<T, D>::RoPE(const ModelConfig& config)
    : cos_sin(precompute_rope_values<D>(config.head_dim, config.rope_theta,
                                        config.max_position_embeddings)){};

template <DType T, Device D> Tensor<T, D> RoPE<T, D>::forward(TensorView<T, D> inputs) const {
  auto cos = std::get<0>(cos_sin);
  auto sin = std::get<1>(cos_sin);

  Shape shape = inputs.shape;
  size_t batch_size = shape[0];
  size_t num_heads = shape[1];
  size_t seq_len = shape[2];
  size_t head_dim = shape[3];

  assert(head_dim % 2 == 0);

  Tensor<float, D> inputs_f32 = inputs.template to<float>(); // all calculations in float32

  // split input into first half and second half, and negate the second half
  auto first_half = slice(inputs_f32.view(), -1, 0, head_dim / 2);
  auto second_half = slice(inputs_f32.view(), -1, head_dim / 2, head_dim);

  second_half = mul(second_half.view(), float(-1.0));

  // adjust sin and cos shapes for broadcasting
  auto adj_cos_ = slice(cos.view(), 0, 0, seq_len);
  auto adj_cos = adj_cos_.view().view_as({1, 1, seq_len, head_dim});
  auto adj_sin_ = slice(sin.view(), 0, 0, seq_len);
  auto adj_sin = adj_sin_.view().view_as({1, 1, seq_len, head_dim});

  // apply the rotary transformation
  auto rotated = cat(second_half.view(), first_half.view(), -1);

  auto input_cos = mul(inputs_f32.view(), adj_cos);

  auto rotated_sin = mul(rotated.view(), adj_sin);

  auto out = add(input_cos.view(), rotated_sin.view());

  return out.view().template to<T>();
}

template <DType T, Device D> TensorView<const float, D> RoPE<T, D>::cos() const {
  return std::get<0>(cos_sin).view();
}

template <DType T, Device D> TensorView<const float, D> RoPE<T, D>::sin() const {
  return std::get<1>(cos_sin).view();
}

template class llama::RoPE<bfloat16, CPU>;
