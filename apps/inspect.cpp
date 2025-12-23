#include <fmt/format.h>

#include <tensor/loader.hpp>

using namespace tensor;

int main(int argc, char* argv[]) {
  const auto* path = "./tests/model";
  if (argc > 1) {
    path = argv[1];
  }

  fmt::println("Loading weights...");
  Loader<bfloat16, CPU> loader{"./tests/model/model.safetensors"};

  loader.inspect();

  return 0;
}
