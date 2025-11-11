#include <forward/loader.hpp>
#include <forward/model.hpp>

#include <fmt/format.h>

int main() {
  model::Model mod("./tests/model");

  loader::inspect_safetensors("./tests/model/model.safetensors");

  auto prompt = "Hello, world";

  auto completion = mod.generate(prompt, 5);

  fmt::print("Prompt: {}", prompt);
  fmt::print("Completion: {}", completion);

  return 0;
}
