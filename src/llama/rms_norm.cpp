#include <cmath>
#include <llama/rms_norm.hpp>

using namespace llama;

void RMSNorm::set_weights(tensor::TensorView<float> weights) {
  weights_ = weights;
}

tensor::Tensor<float>
RMSNorm::forward(tensor::TensorView<float> &inputs) const {
  const size_t batch_size = inputs.shape[0];
  const size_t seq_len = inputs.shape[1];
  const size_t hidden_dim = inputs.shape[2];

  assert(weights_.shape[0] == hidden_dim);

  const auto w = weights_.span();

  tensor::Tensor<float> out_{{batch_size, seq_len, hidden_dim}};

  auto out = out_.view();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      const auto hid_span = inputs.get(b, s).span();
      auto out_span = out.get(b, s).span();

      std::vector<float> buf{};
      buf.reserve(hidden_dim);

      // calculate RMS
      auto sum = 0.0;
      for (int h = 0; h < hidden_dim; ++h) {
        sum += std::pow(hid_span[h], 2);
      }

      auto rms = std::sqrt(sum / hid_span.size());

      // normalize values
      for (int h = 0; h < hidden_dim; ++h) {
        buf[h] = (hid_span[h] / rms) * w[h];
      }

      std::copy_n(std::span(buf).data(), hidden_dim, out_span.data());
    }
  }

  return out_;
}
