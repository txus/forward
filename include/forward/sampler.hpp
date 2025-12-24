#pragma once

#include <forward/tokenizer.hpp>
#include <llama/model.hpp>
#include <tensor/tensor.hpp>

namespace sampler {

struct GreedyConfig {};

template <typename T> struct is_config : std::false_type {};
template <> struct is_config<GreedyConfig> : std::true_type {};

template <typename C>
concept Config = is_config<C>::value;

template <tensor::DType T, tensor::Device D, Config C> struct Sampler {
private:
  C config_;
  tokenizer::Tokenizer* tokenizer_;

  virtual tensor::Tensor<int, D> sample(tensor::Tensor<T, D> logits) = 0;

public:
  explicit Sampler(C config, tokenizer::Tokenizer& tokenizer);
  virtual ~Sampler() = default;

  std::tuple<std::string, float> generate(llama::Model<T, D>& model, std::string_view prompt,
                                          size_t max_num_tokens);
};

template <tensor::DType T, tensor::Device D> struct GreedySampler : Sampler<T, D, GreedyConfig> {
  using Sampler<T, D, GreedyConfig>::Sampler;
  tensor::Tensor<int, D> sample(tensor::Tensor<T, D> logits);
};

} // namespace sampler
