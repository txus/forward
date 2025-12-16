#include <fmt/format.h>

#include <forward/loader.hpp>
#include <forward/sampler.hpp>
#include <forward/tokenizer.hpp>
#include <llama/model.hpp>
#include <tensor/tensor.hpp>

using namespace llama;
using namespace tensor;

int main(int argc, char* argv[]) {
  const auto* path = "./tests/model";
  if (argc > 1) {
    path = argv[1];
  }

  tokenizer::Tokenizer tok("./tests/model/tokenizer.json");

  sampler::GreedySampler<bfloat16, CPU> sampler{sampler::GreedyConfig{}, tok};

  Model<bfloat16, CPU> mod("./tests/model/config.json");

  // loader::inspect_safetensors("./tests/model/model.safetensors");

  fmt::println("Loading weights...");
  auto weights = loader::load_weights<bfloat16, CPU>(
      "./tests/model/model.safetensors",
      // prologue
      "model.embed_tokens.weight",
      // prenorm
      "model.layers.0.input_layernorm.weight", "model.layers.1.input_layernorm.weight",
      "model.layers.2.input_layernorm.weight", "model.layers.3.input_layernorm.weight",
      "model.layers.4.input_layernorm.weight", "model.layers.5.input_layernorm.weight",
      "model.layers.6.input_layernorm.weight", "model.layers.7.input_layernorm.weight",
      "model.layers.8.input_layernorm.weight", "model.layers.9.input_layernorm.weight",
      "model.layers.10.input_layernorm.weight", "model.layers.11.input_layernorm.weight",
      "model.layers.12.input_layernorm.weight", "model.layers.13.input_layernorm.weight",
      "model.layers.14.input_layernorm.weight", "model.layers.15.input_layernorm.weight",
      // postnorm
      "model.layers.0.post_attention_layernorm.weight",
      "model.layers.1.post_attention_layernorm.weight",
      "model.layers.2.post_attention_layernorm.weight",
      "model.layers.3.post_attention_layernorm.weight",
      "model.layers.4.post_attention_layernorm.weight",
      "model.layers.5.post_attention_layernorm.weight",
      "model.layers.6.post_attention_layernorm.weight",
      "model.layers.7.post_attention_layernorm.weight",
      "model.layers.8.post_attention_layernorm.weight",
      "model.layers.9.post_attention_layernorm.weight",
      "model.layers.10.post_attention_layernorm.weight",
      "model.layers.11.post_attention_layernorm.weight",
      "model.layers.12.post_attention_layernorm.weight",
      "model.layers.13.post_attention_layernorm.weight",
      "model.layers.14.post_attention_layernorm.weight",
      "model.layers.15.post_attention_layernorm.weight",
      // q proj
      "model.layers.0.self_attn.q_proj.weight", "model.layers.1.self_attn.q_proj.weight",
      "model.layers.2.self_attn.q_proj.weight", "model.layers.3.self_attn.q_proj.weight",
      "model.layers.4.self_attn.q_proj.weight", "model.layers.5.self_attn.q_proj.weight",
      "model.layers.6.self_attn.q_proj.weight", "model.layers.7.self_attn.q_proj.weight",
      "model.layers.8.self_attn.q_proj.weight", "model.layers.9.self_attn.q_proj.weight",
      "model.layers.10.self_attn.q_proj.weight", "model.layers.11.self_attn.q_proj.weight",
      "model.layers.12.self_attn.q_proj.weight", "model.layers.13.self_attn.q_proj.weight",
      "model.layers.14.self_attn.q_proj.weight", "model.layers.15.self_attn.q_proj.weight",
      // k proj
      "model.layers.0.self_attn.k_proj.weight", "model.layers.1.self_attn.k_proj.weight",
      "model.layers.2.self_attn.k_proj.weight", "model.layers.3.self_attn.k_proj.weight",
      "model.layers.4.self_attn.k_proj.weight", "model.layers.5.self_attn.k_proj.weight",
      "model.layers.6.self_attn.k_proj.weight", "model.layers.7.self_attn.k_proj.weight",
      "model.layers.8.self_attn.k_proj.weight", "model.layers.9.self_attn.k_proj.weight",
      "model.layers.10.self_attn.k_proj.weight", "model.layers.11.self_attn.k_proj.weight",
      "model.layers.12.self_attn.k_proj.weight", "model.layers.13.self_attn.k_proj.weight",
      "model.layers.14.self_attn.k_proj.weight", "model.layers.15.self_attn.k_proj.weight",
      // v proj
      "model.layers.0.self_attn.v_proj.weight", "model.layers.1.self_attn.v_proj.weight",
      "model.layers.2.self_attn.v_proj.weight", "model.layers.3.self_attn.v_proj.weight",
      "model.layers.4.self_attn.v_proj.weight", "model.layers.5.self_attn.v_proj.weight",
      "model.layers.6.self_attn.v_proj.weight", "model.layers.7.self_attn.v_proj.weight",
      "model.layers.8.self_attn.v_proj.weight", "model.layers.9.self_attn.v_proj.weight",
      "model.layers.10.self_attn.v_proj.weight", "model.layers.11.self_attn.v_proj.weight",
      "model.layers.12.self_attn.v_proj.weight", "model.layers.13.self_attn.v_proj.weight",
      "model.layers.14.self_attn.v_proj.weight", "model.layers.15.self_attn.v_proj.weight",
      // o proj
      "model.layers.0.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight",
      "model.layers.2.self_attn.o_proj.weight", "model.layers.3.self_attn.o_proj.weight",
      "model.layers.4.self_attn.o_proj.weight", "model.layers.5.self_attn.o_proj.weight",
      "model.layers.6.self_attn.o_proj.weight", "model.layers.7.self_attn.o_proj.weight",
      "model.layers.8.self_attn.o_proj.weight", "model.layers.9.self_attn.o_proj.weight",
      "model.layers.10.self_attn.o_proj.weight", "model.layers.11.self_attn.o_proj.weight",
      "model.layers.12.self_attn.o_proj.weight", "model.layers.13.self_attn.o_proj.weight",
      "model.layers.14.self_attn.o_proj.weight", "model.layers.15.self_attn.o_proj.weight",
      // MLP up proj
      "model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.up_proj.weight",
      "model.layers.2.mlp.up_proj.weight", "model.layers.3.mlp.up_proj.weight",
      "model.layers.4.mlp.up_proj.weight", "model.layers.5.mlp.up_proj.weight",
      "model.layers.6.mlp.up_proj.weight", "model.layers.7.mlp.up_proj.weight",
      "model.layers.8.mlp.up_proj.weight", "model.layers.9.mlp.up_proj.weight",
      "model.layers.10.mlp.up_proj.weight", "model.layers.11.mlp.up_proj.weight",
      "model.layers.12.mlp.up_proj.weight", "model.layers.13.mlp.up_proj.weight",
      "model.layers.14.mlp.up_proj.weight", "model.layers.15.mlp.up_proj.weight",
      // MLP gate proj
      "model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.gate_proj.weight",
      "model.layers.2.mlp.gate_proj.weight", "model.layers.3.mlp.gate_proj.weight",
      "model.layers.4.mlp.gate_proj.weight", "model.layers.5.mlp.gate_proj.weight",
      "model.layers.6.mlp.gate_proj.weight", "model.layers.7.mlp.gate_proj.weight",
      "model.layers.8.mlp.gate_proj.weight", "model.layers.9.mlp.gate_proj.weight",
      "model.layers.10.mlp.gate_proj.weight", "model.layers.11.mlp.gate_proj.weight",
      "model.layers.12.mlp.gate_proj.weight", "model.layers.13.mlp.gate_proj.weight",
      "model.layers.14.mlp.gate_proj.weight", "model.layers.15.mlp.gate_proj.weight",
      // MLP down proj
      "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight",
      "model.layers.2.mlp.down_proj.weight", "model.layers.3.mlp.down_proj.weight",
      "model.layers.4.mlp.down_proj.weight", "model.layers.5.mlp.down_proj.weight",
      "model.layers.6.mlp.down_proj.weight", "model.layers.7.mlp.down_proj.weight",
      "model.layers.8.mlp.down_proj.weight", "model.layers.9.mlp.down_proj.weight",
      "model.layers.10.mlp.down_proj.weight", "model.layers.11.mlp.down_proj.weight",
      "model.layers.12.mlp.down_proj.weight", "model.layers.13.mlp.down_proj.weight",
      "model.layers.14.mlp.down_proj.weight", "model.layers.15.mlp.down_proj.weight",
      // epilogue
      "model.norm.weight");
  mod.load_weights(weights);

  fmt::println("Weights loaded! Performing inference...");

  std::string prompt = "The capital of france is";

  auto out = sampler.generate(mod, prompt, 2);

  fmt::println("Result: {}", out);

  return 0;
}
