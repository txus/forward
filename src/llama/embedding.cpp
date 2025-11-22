#include <forward/loader.hpp>
#include <llama/embedding.hpp>

using namespace llama;
using namespace tensor;

template <DType T, Device D>
void Embedding<T, D>::set_weights(TensorView<T, D> weights) {
  weights_ = weights;
}

template <>
Tensor<bfloat16, CPU>
Embedding<bfloat16, CPU>::forward(TensorView<int, CPU> &token_ids) const {
  const auto w_shape = weights_.shape;

  const size_t vocab_size = w_shape[0];
  const size_t hidden_dim = w_shape[1];

  const size_t batch_size = token_ids.shape[0];
  const size_t seq_len = token_ids.shape[1];

  Tensor<bfloat16, CPU> out{{batch_size, seq_len, hidden_dim}};

  const auto w_span = weights_.span();
  const auto ids_span = token_ids.span();
  auto out_span = out.span();

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      // take the sth token
      size_t i = b * seq_len + s;
      int tok_id = ids_span[i];
      assert(tok_id >= 0 && static_cast<size_t>(tok_id) < vocab_size);

      size_t src_off = static_cast<size_t>(tok_id) * hidden_dim;
      size_t dest_off = i * hidden_dim;

      std::copy_n(w_span.data() + src_off, hidden_dim,
                  out_span.data() + dest_off);
    }
  }

  return out;
}

template class llama::Embedding<bfloat16, CPU>;
