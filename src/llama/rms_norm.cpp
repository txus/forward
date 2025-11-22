#include <cmath>
#include <llama/rms_norm.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
void RMSNorm<T, D>::set_weights(TensorView<T, D> weights) {
  weights_ = weights;
}

template <>
Tensor<bfloat16, CPU>
RMSNorm<bfloat16, CPU>::forward(TensorView<bfloat16, CPU> &inputs) const {
  const size_t batch_size = inputs.shape[0];
  const size_t seq_len = inputs.shape[1];
  const size_t hidden_dim = inputs.shape[2];

  assert(weights_.shape[0] == hidden_dim);

  const auto w = weights_.span();

  Tensor<bfloat16, CPU> out_{{batch_size, seq_len, hidden_dim}};

  auto out = out_.view();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      const auto hid_span = inputs.get(b, s).span();
      auto out_span = out.get(b, s).span();

      std::vector<bfloat16> buf{};
      buf.reserve(hidden_dim);

      // calculate RMS, accumulate in fp32
      float sum = 0.0;
      for (int h = 0; h < hidden_dim; ++h) {
        sum += std::pow(hid_span[h], 2);
      }

      float rms = std::sqrt(sum / hid_span.size());

      // normalize values
      for (int h = 0; h < hidden_dim; ++h) {
        buf[h] = bfloat16((float(hid_span[h]) / rms) * w[h]);
      }

      std::copy_n(std::span(buf).data(), hidden_dim, out_span.data());
    }
  }

  return out_;
}

template class llama::RMSNorm<bfloat16, CPU>;
