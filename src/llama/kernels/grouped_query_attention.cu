#include <llama/grouped_query_attention.hpp>
#include <tensor/dtype.hpp>
#include <tensor/device_type.hpp>

namespace llama {

template<typename T>
__global__ void gqa_fused(
    T* out, size_t o_b_stride, size_t o_s_stride, size_t o_d_stride,
    T* qs, size_t q_b_stride, size_t q_h_stride, size_t q_s_stride, size_t q_d_stride,
    T* ks, size_t k_b_stride, size_t k_h_stride, size_t k_s_stride, size_t k_d_stride,
    T* vs, size_t v_b_stride, size_t v_h_stride, size_t v_s_stride, size_t v_d_stride,
    T scale_factor, size_t group_size, size_t d_out, bool is_causal) {
  // hehe let's go
}

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> gqa_forward_fused(const TensorView<T, D> &qs,
                                                    const TensorView<T, D> &ks,
                                                    const TensorView<T, D> &vs,
                                                    T scale_factor,
                                                    size_t group_size,
                                                    size_t d_out,
                                                    bool is_causal) {
  Shape shape = qs.shape;
  unsigned int batch_size = shape[0];
  unsigned int num_q_heads = shape[1];
  unsigned int num_k_heads = ks.shape[1];
  unsigned int seq_len = shape[2];
  unsigned int head_dim = shape[3];

  size_t q_b_stride = qs.stride[0]; size_t q_h_stride = qs.stride[1];
  size_t q_s_stride = qs.stride[2]; size_t q_d_stride = qs.stride[4];

  size_t k_b_stride = ks.stride[0]; size_t k_h_stride = ks.stride[1];
  size_t k_s_stride = ks.stride[2]; size_t k_d_stride = ks.stride[4];

  size_t v_b_stride = vs.stride[0]; size_t v_h_stride = vs.stride[1];
  size_t v_s_stride = vs.stride[2]; size_t v_d_stride = vs.stride[4];

  size_t o_b_stride = (seq_len * head_dim);
  size_t o_s_stride = head_dim;
  size_t o_d_stride = 1;

  auto n_elements = static_cast<size_t>(batch_size * seq_len) * head_dim;

  Shape out_shape{batch_size, seq_len, head_dim * num_q_heads};

  TensorStorage<T, CUDA> storage(n_elements);
  Tensor<T, CUDA> out{out_shape, std::move(storage)};

  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* q_d = reinterpret_cast<Cuda<T>*>(qs.data); // NOLINT
  auto* k_d = reinterpret_cast<Cuda<T>*>(ks.data); // NOLINT
  auto* v_d = reinterpret_cast<Cuda<T>*>(vs.data); // NOLINT

  Cuda<T> scale_factor_d = to_device_type(scale_factor, CUDA{});

  unsigned int grid_size = 1;
  unsigned int block_size = 1;

  gqa_fused<Cuda<T>><<<grid_size, block_size>>>(
      out_d, o_b_stride, o_s_stride, o_d_stride,
      q_d, q_b_stride, q_h_stride, q_s_stride, q_d_stride,
      k_d, k_b_stride, k_h_stride, k_s_stride, k_d_stride,
      v_d, v_b_stride, v_h_stride, v_s_stride, v_d_stride,
      scale_factor_d, group_size, d_out, is_causal
      );

  return out;
}

template
Tensor<bfloat16, CUDA>
gqa_forward_fused(const TensorView<bfloat16, CUDA>& qs,
                  const TensorView<bfloat16, CUDA>& ks,
                  const TensorView<bfloat16, CUDA>& vs, bfloat16 scale_factor,
                  size_t group_size, size_t d_out, bool is_causal);

}
