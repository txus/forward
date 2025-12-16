#include <llama/mlp.hpp>
#include <nn/act.hpp>
#include <tensor/ops.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D> MLP<T, D>::MLP(const ModelConfig& config) {
  act_fn = nn::make_activation(config.hidden_act);
}

template <DType T, Device D>
void MLP<T, D>::load_weights(std::unordered_map<std::string, Tensor<T, D> /*unused*/>& weight_map,
                             size_t layer_idx) {
  up_proj.set_weights(
      weight_map.at(fmt::format("model.layers.{}.mlp.up_proj.weight", layer_idx)).view(), true);
  gate_proj.set_weights(
      weight_map.at(fmt::format("model.layers.{}.mlp.gate_proj.weight", layer_idx)).view(), true);
  down_proj.set_weights(
      weight_map.at(fmt::format("model.layers.{}.mlp.down_proj.weight", layer_idx)).view(), true);
}

template <DType T, Device D> Tensor<T, D> MLP<T, D>::forward(TensorView<T, D> inputs) const {
  auto up_proj_t = up_proj.forward(inputs);
  auto up_proj_v = up_proj_t.view();

  auto gate_proj_t = gate_proj.forward(inputs);
  auto gate_proj_v = gate_proj_t.view();

  auto activated_t = std::visit([&](auto& function) { return function(gate_proj_v); }, act_fn);
  auto activated_v = activated_t.view();

  auto gated_t = mul<T, D>(activated_v, up_proj_v);
  auto gated_v = gated_t.view();

  auto down_proj_t = down_proj.forward(gated_v);

  return down_proj_t;
}

template class llama::MLP<bfloat16, CPU>;
