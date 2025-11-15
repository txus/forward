#include <forward/loader.hpp>
#include <forward/model.hpp>
#include <forward/nn/embedding.hpp>
#include <forward/tensor.hpp>

namespace model {

Model::Model(std::string_view model_path)
    : tok(std::string(model_path) + "/tokenizer.json"),
      embed(loader::load_from_safetensors(std::string(model_path) +
                                              "/model.safetensors",
                                          "model.embed_tokens.weight")) {}

Model::~Model() = default;

tensor::Tensor<float> Model::forward(tensor::Tensor<int> token_ids) {
  auto embedded = embed.forward(token_ids);

  return embedded;
}
std::string Model::generate(std::string_view prompt, uint16_t max_tokens) {
  auto encoded = tok.encode(prompt);
  auto input = tensor::Tensor<int>{{1, encoded.size()}, std::move(encoded)};

  auto out = forward(input);

  std::println("Out shape: {}", out.shape());
  std::println("Out raw: {}", out.raw());

  return "completion";
}

} // namespace model
