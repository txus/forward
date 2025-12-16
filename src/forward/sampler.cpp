#include <forward/sampler.hpp>
#include <forward/tokenizer.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace tensor;
using namespace tokenizer;

namespace sampler {

template <tensor::DType T, tensor::Device D, Config C>
Sampler<T, D, C>::Sampler(C config, tokenizer::Tokenizer& tokenizer)
    : config_(config), tokenizer_(&tokenizer){};

template <DType T, Device D, Config C>
std::string Sampler<T, D, C>::generate(llama::Model<T, D> model, std::string_view prompt,
                                       size_t max_num_tokens) {
  fmt::println("Prompt: {}", prompt);
  std::vector<int> token_ids = tokenizer_->encode(prompt);

  std::vector<int> sampled_token_ids;
  sampled_token_ids.reserve(max_num_tokens);

  for (size_t remaining_tokens = max_num_tokens; remaining_tokens > 0; --remaining_tokens) {
    Tensor<int, D> inputs({1, token_ids.size()}, std::vector<int>(token_ids));

    fmt::println("Tokenized input: {}", inputs.view());

    auto logits = model.forward(inputs.view());
    // logits is [batch_size, seq_len, vocab_size], we want the logits for the last
    // token in the sequence
    auto seq_len = logits.shape()[1];

    // [batch_size, vocab_size]
    auto last_token_logits = slice(logits.view(), 1, seq_len - 1, seq_len);

    auto sampled_ids = sample(last_token_logits);
    auto sampled_span = sampled_ids.span();
    for (auto tok_id : sampled_span) {
      fmt::println("token: {}", tok_id);
      token_ids.push_back(tok_id);
      fmt::println("{}", tokenizer_->decode(token_ids));
      sampled_token_ids.push_back(tok_id);
    }
  }

  return tokenizer_->decode(sampled_token_ids);
}

template <DType T, Device D>
Tensor<int, D> GreedySampler<T, D>::sample(tensor::Tensor<T, D> logits) {
  return argmax(logits.view(), -1, false);
}

} // namespace sampler

template class sampler::Sampler<bfloat16, CPU, sampler::GreedyConfig>;
template class sampler::GreedySampler<bfloat16, CPU>;
