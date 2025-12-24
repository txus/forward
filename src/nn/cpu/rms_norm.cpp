#include <cmath>
#include <nn/rms_norm.hpp>
#include <tensor/loader.hpp>

using namespace nn;
using namespace tensor;

template <DType T, Device D> RMSNorm<T, D>::RMSNorm(float eps) : eps(eps){};

template <DType T, Device D>
void RMSNorm<T, D>::load_weights(const tensor::Loader<T, D>& loader, std::string_view name) {
  weights_ = loader.load(name);
}

template <DType T, Device D>
void RMSNorm<T, D>::load_weights(tensor::Tensor<const T, D> weights) {
  weights_ = std::move(weights);
}

template <DType T, Device D>
void RMSNorm<T, D>::load_weights(const tensor::Tensor<T, D>& weights) {
  auto storage = TensorStorage<const T, D>::borrow(weights.data(), weights.size());
  weights_ = Tensor<const T, D>{weights.shape(), std::move(storage)};
}

template <DType T, Device D>
Tensor<T, D> RMSNorm<T, D>::forward(const TensorView<T, D>& inputs) const {
  auto weights = weights_.view();

  const size_t batch_size = inputs.shape[0];
  const size_t seq_len = inputs.shape[1];
  const size_t hidden_dim = inputs.shape[2];

  assert(weights.shape[0] == hidden_dim);

  const auto w_span = weights.span();

  Tensor<T, D> out_{{batch_size, seq_len, hidden_dim}};

  auto out = out_.view();

  for (int row_idx = 0; row_idx < batch_size; ++row_idx) {
    for (int seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
      const auto hid_span = inputs.get(row_idx, seq_pos).span();
      auto out_span = out.get(row_idx, seq_pos).span();

      std::vector<T> buf{};
      buf.reserve(hidden_dim);

      // calculate RMS, accumulate in fp32
      float sum = 0.0;
      for (int channel_idx = 0; channel_idx < hidden_dim; ++channel_idx) {
        auto val = static_cast<float>(hid_span[channel_idx]);
        sum += val * val;
      }

      float rms = 1.0 / std::sqrt(eps + (sum / hid_span.size())); // NOLINT(*-narrowing-conversions)

      // normalize values
      for (int channel_idx = 0; channel_idx < hidden_dim; ++channel_idx) {
        buf[channel_idx] = T(float(hid_span[channel_idx]) * rms) * w_span[channel_idx];
      }

      std::copy_n(std::span(buf).data(), hidden_dim, out_span.data());
    }
  }

  return out_;
}

template class nn::RMSNorm<bfloat16, CPU>;
