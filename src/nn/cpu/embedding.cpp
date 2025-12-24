#include <nn/embedding.hpp>
#include <tensor/loader.hpp>

using namespace nn;
using namespace tensor;

template <DType T, Device D>
void Embedding<T, D>::load_weights(const tensor::Loader<T, D>& loader) {
  weights_ = loader.load("model.embed_tokens.weight");
}

template <DType T, Device D>
void Embedding<T, D>::load_weights(tensor::Tensor<const T, D> weights) {
  weights_ = std::move(weights);
}

template <DType T, Device D>
void Embedding<T, D>::load_weights(const tensor::Tensor<T, D>& weights) {
  auto storage = TensorStorage<const T, D>::borrow(weights.data(), weights.size());
  weights_ = Tensor<const T, D>{weights.shape(), std::move(storage)};
}

template <DType T, Device D>
Tensor<T, D> Embedding<T, D>::forward(const TensorView<int, D>& token_ids) const {
  auto weights = weights_.view();
  const auto w_shape = weights.shape;

  const size_t vocab_size = w_shape[0];
  const size_t hidden_dim = w_shape[1];

  const size_t batch_size = token_ids.shape[0];
  const size_t seq_len = token_ids.shape[1];

  Tensor<T, D> out{{batch_size, seq_len, hidden_dim}};

  const auto w_span = weights.span();
  const auto ids_span = token_ids.span();
  auto out_span = out.span();

  for (size_t row_idx = 0; row_idx < batch_size; ++row_idx) {
    for (size_t seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
      // take the sth token
      size_t idx = (row_idx * seq_len) + seq_pos;
      int tok_id = ids_span[idx];
      assert(tok_id >= 0 && static_cast<size_t>(tok_id) < vocab_size);

      size_t src_off = static_cast<size_t>(tok_id) * hidden_dim;
      size_t dest_off = idx * hidden_dim;

      std::copy_n(w_span.data() + src_off, hidden_dim, out_span.data() + dest_off);
    }
  }

  return out;
}

template class nn::Embedding<bfloat16, CPU>;
