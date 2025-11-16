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
                                      "model.embed_tokens.weight");
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
