#include <forward/loader.hpp>
#include <forward/tokenizer.hpp>
#include <llama/model.hpp>
#include <tensor/tensor.hpp>

#include <fmt/format.h>

int main(int argc, char *argv[]) {
  auto path = "./tests/model";
  if (argc > 1) {
    path = argv[1];
  }

  tokenizer::Tokenizer tok("./tests/model/tokenizer.json");

  llama::Model mod("./tests/model/config.json");

  // loader::inspect_safetensors("./tests/model/model.safetensors");

  auto weights = loader::load_weights("./tests/model/model.safetensors",
                                      "model.embed_tokens.weight",
                                      "model.layers.0.input_layernorm.weight",
                                      "model.layers.1.input_layernorm.weight",
                                      "model.layers.2.input_layernorm.weight",
                                      "model.layers.3.input_layernorm.weight",
                                      "model.layers.4.input_layernorm.weight",
                                      "model.layers.5.input_layernorm.weight",
                                      "model.layers.6.input_layernorm.weight",
                                      "model.layers.7.input_layernorm.weight",
                                      "model.layers.8.input_layernorm.weight",
                                      "model.layers.9.input_layernorm.weight",
                                      "model.layers.10.input_layernorm.weight",
                                      "model.layers.11.input_layernorm.weight",
                                      "model.layers.12.input_layernorm.weight",
                                      "model.layers.13.input_layernorm.weight",
                                      "model.layers.14.input_layernorm.weight",
                                      "model.layers.15.input_layernorm.weight");
  mod.load_weights(weights);

  auto prompt = "Hello, world";
  auto token_ids = tok.encode(prompt);

  auto input_ids_ =
      tensor::Tensor<int>{{1, token_ids.size()}, std::move(token_ids)};

  auto input_ids = input_ids_.view();

  auto result = mod.forward(input_ids);

  fmt::println("Result: {}", result.view());

  return 0;
}
