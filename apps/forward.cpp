#include <fmt/format.h>

#include <forward/sampler.hpp>
#include <forward/tokenizer.hpp>
#include <llama/model.hpp>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

using namespace llama;
using namespace tensor;

int main(int argc, char* argv[]) {
  const auto* path = "./tests/model";
  if (argc > 1) {
    path = argv[1];
  }

  size_t max_tokens = 128;
  size_t kv_cache_size = max_tokens;

  tokenizer::Tokenizer tok("./tests/model/tokenizer.json");

  sampler::GreedySampler<bfloat16, CPU> sampler{sampler::GreedyConfig{}, tok};

  Model<bfloat16, CPU> mod("./tests/model/config.json", max_tokens, kv_cache_size);

  // loader::inspect_safetensors("./tests/model/model.safetensors");

  fmt::println("Loading weights...");
  Loader<bfloat16, CPU> loader{"./tests/model/model.safetensors"};
  mod.load_weights(loader);

  fmt::println("Weights loaded! Performing inference...");

  std::string prompt = "The capital of france is";

  fmt::println("Prompt: {}", prompt);

  auto gen_and_tok_s = sampler.generate(mod, prompt, 12);

  auto out = std::get<0>(gen_and_tok_s);
  auto tok_s = std::get<1>(gen_and_tok_s);

  out = fmt::format(fmt::fg(fmt::color::aqua), "{}", out);

  fmt::println("{}{}", prompt, out);

  fmt::println("Tokens / sec: {}", tok_s);

  return 0;
}
