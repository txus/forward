#include <forward/model.hpp>

namespace model {

Model::Model(std::string_view model_path)
    : tok(std::string(model_path) + "/tokenizer.json") {}

Model::~Model() = default;

std::vector<double> Model::forward(std::vector<int> token_ids) {
  return {2.4, 8.5};
}
std::string Model::generate(std::string_view prompt, uint16_t max_tokens) {
  return "completion";
}

} // namespace model
