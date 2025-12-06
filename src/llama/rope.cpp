#include <llama/rope.hpp>
#include <tensor/ops.hpp>
#include <tuple>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
inline std::tuple<Tensor<T, D>, Tensor<T, D>>
precompute_rope_values(size_t head_dim, float theta_base, size_t context_length) {
  assert(head_dim % 2 == 0);

  // compute the inverse frequencies
  Tensor<int, D> range = arange<D>(0, head_dim, 2);
  auto range_float = range.view().to_float();
  auto scaled = range_float.view() / float(head_dim);
  auto powd = pow(theta_base, scaled.view());
  auto inv_freq_ = pow(powd.view(), float(-1.0));

  auto positions = arange<D>(0.0, float(context_length), 1.0);
  auto lhs = positions.view().view_as({positions.shape()[0], 1});
  auto rhs = inv_freq_.view().view_as({1, inv_freq_.shape()[0]});

  auto angles = matmul(lhs, rhs);

  auto sin = Tensor<T, D>{{3}};
  auto cos = Tensor<T, D>{{3}};
  return std::tuple<Tensor<T, D>, Tensor<T, D>>(sin, cos);
}

template <DType T, Device D>
RoPE<T, D>::RoPE(const ModelConfig& config)
    : cos_sin(precompute_rope_values<T, D>(config.head_dim, config.rope_theta,
                                           config.max_position_embeddings)){};

template <DType T, Device D> Tensor<T, D> RoPE<T, D>::forward(TensorView<T, D> inputs) const {
  return Tensor<T, D>{{3}};
}

template <DType T, Device D> TensorView<const T, D> RoPE<T, D>::cos() const {
  return std::get<0>(cos_sin).view();
}

template <DType T, Device D> TensorView<const T, D> RoPE<T, D>::sin() const {
  return std::get<1>(cos_sin).view();
}

template class llama::RoPE<bfloat16, CPU>;
