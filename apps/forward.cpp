#include <forward/loader.hpp>

#include <fmt/format.h>

int main(int argc, char *argv[]) {
  auto path = "./tests/model";
  if (argc > 1) {
    path = argv[1];
  }

  // model::Model mod(path);

  loader::inspect_safetensors("./tests/model/model.safetensors");

  auto prompt = "Hello, world";

  // auto completion = mod.generate(prompt, 5);

  fmt::print("Prompt: {}", prompt);
  // fmt::print("Completion: {}", completion);

  return 0;
}
