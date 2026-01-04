#include <llama/mlp.hpp>
#include <nn/act.hpp>
#include <tensor/ops.hpp>
#include <util/nvtx.hpp>

using namespace llama;
using namespace tensor;

template <typename T, typename D> MLP<T, D>::MLP(const ModelConfig& config) {
  act_fn = nn::make_activation(config.hidden_act);
}

template <typename T, typename D>
void MLP<T, D>::load_weights(const tensor::Loader<T, D>& loader, size_t layer_idx) {
  up_proj.load_weights(loader, fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx));
  gate_proj.load_weights(loader, fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx));
  down_proj.load_weights(loader, fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx));
}

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> MLP<T, D>::forward(TensorView<T, D> inputs) {
  NVTX_RANGE("mlp");
  auto up_proj_t = up_proj.forward(inputs);
  auto up_proj_v = up_proj_t.view();

  // fmt::println("mlp.up_proj {}", up_proj_v);

  auto gate_proj_t = gate_proj.forward(inputs);
  auto gate_proj_v = gate_proj_t.view();

  // fmt::println("mlp.gate_proj {}", gate_proj_v);

  auto activated_t = std::visit([&](auto& function) { return function(gate_proj_v); }, act_fn);
  auto activated_v = activated_t.view();

  // fmt::println("mlp.act_fn {}", activated_v);

  auto gated_t = mul<T, D>(activated_v, up_proj_v);
  auto gated_v = gated_t.view();

  auto down_proj_t = down_proj.forward(gated_v);

  // fmt::println("mlp.down_proj {}", down_proj_t.view());

  return down_proj_t;
}

template class llama::MLP<bfloat16, CPU>;

#ifdef BACKEND_CUDA
template class llama::MLP<bfloat16, CUDA>;
#endif
