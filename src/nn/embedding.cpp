#include <forward/nn/embedding.hpp>

namespace nn {

Embedding::Embedding(tensor::Tensor<float> weights) : weights(weights) {}
Embedding::~Embedding() = default;

tensor::Tensor<float> Embedding::forward(tensor::Tensor<int> token_ids) {
  size_t hidden_dim = weights.shape()[1];
  size_t batch_size = token_ids.shape()[0];
  size_t seq_len = token_ids.shape()[1];

  tensor::Tensor<float> out{{batch_size, seq_len, hidden_dim}};
  out.fill_(0.0);

  int seq_start = 0;
  for (auto &tok_id : token_ids.raw()) {
    auto token_hid_dim = weights.slice(tok_id);
    for (int hid_dim = 0; hid_dim < hidden_dim; ++hid_dim) {

      out.set_(hid_dim + seq_start, token_hid_dim.at(hid_dim));
    }
    seq_start += hidden_dim;
  }

  return out;
}
} // namespace nn
