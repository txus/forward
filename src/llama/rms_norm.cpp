#include <cmath>
#include <llama/rms_norm.hpp>
#include <utility>

using namespace llama;
using namespace tensor;

template <DType T, Device D> void RMSNorm<T, D>::set_weights(TensorView<T, D> weights) {
  weights_ = std::move(weights);
}

template <>
Tensor<bfloat16, CPU> RMSNorm<bfloat16, CPU>::forward(TensorView<bfloat16, CPU>& inputs) const {
  const size_t batch_size = inputs.shape[0];
  const size_t seq_len = inputs.shape[1];
  const size_t hidden_dim = inputs.shape[2];

  assert(weights_.shape[0] == hidden_dim);

  const auto w_span = weights_.span();

  Tensor<bfloat16, CPU> out_{{batch_size, seq_len, hidden_dim}};

  auto out = out_.view();

  for (int row_idx = 0; row_idx < batch_size; ++row_idx) {
    for (int seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
      const auto hid_span = inputs.get(row_idx, seq_pos).span();
      auto out_span = out.get(row_idx, seq_pos).span();

      std::vector<bfloat16> buf{};
      buf.reserve(hidden_dim);

      // calculate RMS, accumulate in fp32
      float sum = 0.0;
      for (int channel_idx = 0; channel_idx < hidden_dim; ++channel_idx) {
        sum += std::pow(hid_span[channel_idx], 2);
      }

      float rms = std::sqrt(sum / hid_span.size()); // NOLINT(*-narrowing-conversions)

      // normalize values
      for (int channel_idx = 0; channel_idx < hidden_dim; ++channel_idx) {
        buf[channel_idx] = bfloat16((float(hid_span[channel_idx]) / rms) * w_span[channel_idx]);
      }

      std::copy_n(std::span(buf).data(), hidden_dim, out_span.data());
    }
  }

  return out_;
}

template class llama::RMSNorm<bfloat16, CPU>;
