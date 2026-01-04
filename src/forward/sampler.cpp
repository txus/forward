#include <chrono>
#include <forward/sampler.hpp>
#include <forward/tokenizer.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>
#include <util/nvtx.hpp>

using namespace tensor;
using namespace tokenizer;

namespace sampler {

template <tensor::DType T, tensor::Device D, Config C>
Sampler<T, D, C>::Sampler(C config, tokenizer::Tokenizer& tokenizer)
    : config_(config), tokenizer_(&tokenizer){};

template <DType T, Device D, Config C>
std::tuple<std::string, GenerationStats> Sampler<T, D, C>::generate(llama::Model<T, D>& model,
                                                                    std::string_view prompt,
                                                                    size_t max_num_tokens) {
  NVTX_RANGE("generate");

  using std::chrono::duration;
  using std::chrono::high_resolution_clock;

  fmt::println("Prompt: {}", prompt);
  std::vector<int> token_ids = tokenizer_->encode(prompt);

  std::vector<int> sampled_token_ids;
  sampled_token_ids.reserve(max_num_tokens);

  auto start_time = high_resolution_clock::now();
  high_resolution_clock::time_point first_token_time;
  high_resolution_clock::time_point prev_token_time;
  float total_itl_ms = 0.0f;

  for (size_t token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
    // Create input tensor on CPU, then transfer to device if needed
    Tensor<int, device::CPU> inputs_cpu({1, token_ids.size()}, std::vector<int>(token_ids));

    Tensor<int, D> inputs = [&]() {
#ifdef BACKEND_CUDA
      if constexpr (std::same_as<D, device::CUDA>) {
        return inputs_cpu.cuda();
      } else
#endif
      {
        return std::move(inputs_cpu);
      }
    }();

    auto logits = model.forward(inputs.view());
    // logits is [batch_size, seq_len, vocab_size], we want the logits for the last
    // token in the sequence
    auto seq_len = logits.shape()[1];

    // [batch_size, vocab_size]
    auto last_token_logits = slice(logits.view(), 1, seq_len - 1, seq_len);

    auto sampled_ids = sample(std::move(last_token_logits));

    // Transfer sampled ids to CPU to read values
    Tensor<int, device::CPU> sampled_ids_cpu = [&]() {
#ifdef BACKEND_CUDA
      if constexpr (std::same_as<D, device::CUDA>) {
        return sampled_ids.cpu();
      } else
#endif
      {
        return std::move(sampled_ids);
      }
    }();

    auto now = high_resolution_clock::now();

    if (token_idx == 0) {
      first_token_time = now;
    } else {
      duration<float, std::milli> itl = now - prev_token_time;
      total_itl_ms += itl.count();
    }
    prev_token_time = now;

    auto sampled_span = sampled_ids_cpu.span();
    for (auto tok_id : sampled_span) {
      token_ids = std::vector<int>{tok_id};
      sampled_token_ids.push_back(tok_id);
    }
  }

  auto end_time = high_resolution_clock::now();

  // Calculate stats
  duration<float> total_elapsed = end_time - start_time;
  duration<float, std::milli> ttft = first_token_time - start_time;

  GenerationStats stats{
      .tokens_per_sec = float(max_num_tokens) / total_elapsed.count(),
      .ttft_ms = ttft.count(),
      .avg_itl_ms = (max_num_tokens > 1) ? total_itl_ms / float(max_num_tokens - 1) : 0.0f,
  };

  return {tokenizer_->decode(sampled_token_ids), stats};
}

template <DType T, Device D>
Tensor<int, D> GreedySampler<T, D>::sample(tensor::Tensor<T, D> logits) {
  NVTX_RANGE("sample");
  return argmax(logits.view(), -1, false);
}

} // namespace sampler

template class sampler::Sampler<bfloat16, CPU, sampler::GreedyConfig>;
template class sampler::GreedySampler<bfloat16, CPU>;

#ifdef BACKEND_CUDA
template class sampler::Sampler<bfloat16, CUDA, sampler::GreedyConfig>;
template class sampler::GreedySampler<bfloat16, CUDA>;
#endif
