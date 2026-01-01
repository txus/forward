#include <fmt/format.h>
#include <cstring>

#include <forward/sampler.hpp>
#include <forward/tokenizer.hpp>
#include <llama/model.hpp>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

using namespace llama;
using namespace tensor;

template <Device D>
void run_inference(const char* path, tokenizer::Tokenizer& tok) {
  size_t max_tokens = 128;
  size_t kv_cache_size = max_tokens;

  sampler::GreedySampler<bfloat16, D> sampler{sampler::GreedyConfig{}, tok};

  Model<bfloat16, D> mod(fmt::format("{}/config.json", path), max_tokens, kv_cache_size);

  fmt::println("Loading weights...");
  Loader<bfloat16, D> loader{fmt::format("{}/model.safetensors", path)};
  mod.load_weights(loader);

  fmt::println("Weights loaded! Performing inference...");

  std::string prompt = "The capital of france is";

  fmt::println("Prompt: {}", prompt);

  auto [out, stats] = sampler.generate(mod, prompt, 12);

  auto colored_out = fmt::format(fmt::fg(fmt::color::aqua), "{}", out);

  fmt::println("{}{}", prompt, colored_out);

  fmt::println("");
  fmt::println("TTFT:         {:.2f} ms", stats.ttft_ms);
  fmt::println("Avg ITL:      {:.2f} ms", stats.avg_itl_ms);
  fmt::println("Tokens / sec: {:.2f}", stats.tokens_per_sec);
}

int main(int argc, char* argv[]) {
  const char* path = "./tests/model";
  bool use_cuda = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--cuda") == 0) {
      use_cuda = true;
    } else {
      path = argv[i];
    }
  }

  tokenizer::Tokenizer tok(fmt::format("{}/tokenizer.json", path));

  if (use_cuda) {
#ifdef BACKEND_CUDA
    fmt::println("Using CUDA backend");
    run_inference<CUDA>(path, tok);
#else
    fmt::println("Error: CUDA backend not available. Rebuild with CUDA support.");
    return 1;
#endif
  } else {
    fmt::println("Using CPU backend");
    run_inference<CPU>(path, tok);
  }

  return 0;
}
