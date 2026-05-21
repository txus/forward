#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include <cmath>
#include <stdexcept>

#include <llama/grouped_query_attention.hpp>
#include <llama/rope.hpp>
#include <llama/model.hpp>
#include <tensor/device_type.hpp>
#include <tensor/dtype.hpp>
#include <nn/softmax.hpp>

using namespace llama;
using namespace tensor;
using namespace benchmark;

namespace {

constexpr int HEAD_DIM = 64;

inline void cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(fmt::format("{}: {}", msg, cudaGetErrorString(err)));
  }
}

template <typename T>
void fill_cuda_tensor(Tensor<T, CUDA>& t, T value) {
  t.fill_(value);
}

int64_t attention_flops(
    int64_t batch,
    int64_t q_heads,
    int64_t q_seq,
    int64_t kv_seq,
    int64_t head_dim) {
  // QK^T + P@V. Softmax/mask ignored.
  return 4LL * batch * q_heads * q_seq * kv_seq * head_dim;
}

int64_t attention_bytes_rough(
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t q_seq,
    int64_t kv_seq,
    int64_t head_dim,
    int64_t bytes_per_elem) {
  int64_t q_bytes = batch * q_heads * q_seq * head_dim * bytes_per_elem;
  int64_t k_bytes = batch * kv_heads * kv_seq * head_dim * bytes_per_elem;
  int64_t v_bytes = batch * kv_heads * kv_seq * head_dim * bytes_per_elem;
  int64_t o_bytes = batch * q_seq * q_heads * head_dim * bytes_per_elem;
  return q_bytes + k_bytes + v_bytes + o_bytes;
}

template <typename T>
struct GQABenchInputs {
  Tensor<T, CUDA> q;
  Tensor<T, CUDA> k;
  Tensor<T, CUDA> v;

  GQABenchInputs(
      int batch,
      int q_heads,
      int kv_heads,
      int q_seq,
      int kv_seq)
      : q({
            static_cast<size_t>(batch),
            static_cast<size_t>(q_heads),
            static_cast<size_t>(q_seq),
            static_cast<size_t>(HEAD_DIM),
        }),
        k({
            static_cast<size_t>(batch),
            static_cast<size_t>(kv_heads),
            static_cast<size_t>(kv_seq),
            static_cast<size_t>(HEAD_DIM),
        }),
        v({
            static_cast<size_t>(batch),
            static_cast<size_t>(kv_heads),
            static_cast<size_t>(kv_seq),
            static_cast<size_t>(HEAD_DIM),
        }) {
    fill_cuda_tensor(q, T(0.01f));
    fill_cuda_tensor(k, T(0.02f));
    fill_cuda_tensor(v, T(0.03f));
    cuda_check(cudaDeviceSynchronize(), "input fill sync");
  }
};

template <typename T>
void set_common_counters(
    benchmark::State& state,
    int batch,
    int q_heads,
    int kv_heads,
    int q_seq,
    int kv_seq) {
  const int64_t flops_per_iter =
      attention_flops(batch, q_heads, q_seq, kv_seq, HEAD_DIM);

  const int64_t bytes_per_iter =
      attention_bytes_rough(batch, q_heads, kv_heads, q_seq, kv_seq, HEAD_DIM, 2);

  state.counters["FLOP/s"] = benchmark::Counter(
      flops_per_iter,
      benchmark::Counter::kIsIterationInvariantRate);

  state.counters["TFLOP/s"] = benchmark::Counter(
      static_cast<double>(flops_per_iter) / 1.0e12,
      benchmark::Counter::kIsIterationInvariantRate);

  state.counters["Bytes/s"] = benchmark::Counter(
      bytes_per_iter,
      benchmark::Counter::kIsIterationInvariantRate);

  state.SetBytesProcessed(state.iterations() * bytes_per_iter);
}

static void BM_Llama_GQA_Core_Fused_Prefill(benchmark::State& state) {
  const int batch = static_cast<int>(state.range(0));
  const int q_heads = static_cast<int>(state.range(1));
  const int kv_heads = static_cast<int>(state.range(2));
  const int seq_len = static_cast<int>(state.range(3));

  const int group_size = q_heads / kv_heads;
  const int d_out = q_heads * HEAD_DIM;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  GQABenchInputs<bfloat16> inputs(batch, q_heads, kv_heads, seq_len, seq_len);

  for (int i = 0; i < 10; ++i) {
    auto out = gqa_forward_fused<bfloat16, CUDA, HEAD_DIM>(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        scale,
        group_size,
        d_out,
        /*position_offset=*/0,
        /*is_causal=*/true);

    benchmark::DoNotOptimize(out.data());
  }
  cuda_check(cudaDeviceSynchronize(), "fused prefill warmup sync");

  for (auto _ : state) {
    auto out = gqa_forward_fused<bfloat16, CUDA, HEAD_DIM>(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        scale,
        group_size,
        d_out,
        /*position_offset=*/0,
        /*is_causal=*/true);

    benchmark::DoNotOptimize(out.data());
    cuda_check(cudaDeviceSynchronize(), "fused prefill benchmark sync");
  }

  set_common_counters<bfloat16>(
      state, batch, q_heads, kv_heads, seq_len, seq_len);
}

static void BM_Llama_GQA_Core_NonFused_Prefill(benchmark::State& state) {
  const int batch = static_cast<int>(state.range(0));
  const int q_heads = static_cast<int>(state.range(1));
  const int kv_heads = static_cast<int>(state.range(2));
  const int seq_len = static_cast<int>(state.range(3));

  const int group_size = q_heads / kv_heads;
  const int d_out = q_heads * HEAD_DIM;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  GQABenchInputs<bfloat16> inputs(batch, q_heads, kv_heads, seq_len, seq_len);

  nn::Softmax softmax;

  // Create mask like model.cpp does - size is max_tokens (here seq_len for prefill)
  auto attn_mask = causal_attention_mask<int, CUDA>(seq_len);

  // Slice like grouped_query_attention.cpp does for no-cache case:
  // slice(attn_mask, 0, 0, input_seq_len) then slice(..., 1, 0, input_seq_len)
  auto mask_to_use = slice(attn_mask.view(), 0, 0, seq_len);
  mask_to_use = slice(mask_to_use.view(), 1, 0, seq_len);
  auto attention_mask = mask_to_use.view();

  for (int i = 0; i < 10; ++i) {
    auto out = gqa_forward(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        attention_mask,
        softmax,
        scale,
        group_size,
        d_out);

    benchmark::DoNotOptimize(out.data());
  }
  cuda_check(cudaDeviceSynchronize(), "non-fused prefill warmup sync");

  for (auto _ : state) {
    auto out = gqa_forward(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        attention_mask,
        softmax,
        scale,
        group_size,
        d_out);

    benchmark::DoNotOptimize(out.data());
    cuda_check(cudaDeviceSynchronize(), "non-fused prefill benchmark sync");
  }

  set_common_counters<bfloat16>(
      state, batch, q_heads, kv_heads, seq_len, seq_len);
}


static void BM_Llama_GQA_Core_Fused_Decode(benchmark::State& state) {
  const int batch = static_cast<int>(state.range(0));
  const int q_heads = static_cast<int>(state.range(1));
  const int kv_heads = static_cast<int>(state.range(2));
  const int q_seq = static_cast<int>(state.range(3));
  const int kv_seq = static_cast<int>(state.range(4));

  const int group_size = q_heads / kv_heads;
  const int d_out = q_heads * HEAD_DIM;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  const int position_offset = kv_seq - q_seq;

  GQABenchInputs<bfloat16> inputs(batch, q_heads, kv_heads, q_seq, kv_seq);

  for (int i = 0; i < 20; ++i) {
    auto out = gqa_forward_fused<bfloat16, CUDA, HEAD_DIM>(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        scale,
        group_size,
        d_out,
        position_offset,
        /*is_causal=*/true);

    benchmark::DoNotOptimize(out.data());
  }
  cuda_check(cudaDeviceSynchronize(), "fused decode warmup sync");

  for (auto _ : state) {
    auto out = gqa_forward_fused<bfloat16, CUDA, HEAD_DIM>(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        scale,
        group_size,
        d_out,
        position_offset,
        /*is_causal=*/true);

    benchmark::DoNotOptimize(out.data());
    cuda_check(cudaDeviceSynchronize(), "fused decode benchmark sync");
  }

  set_common_counters<bfloat16>(
      state, batch, q_heads, kv_heads, q_seq, kv_seq);
}


static void BM_Llama_GQA_Core_NonFused_Decode(benchmark::State& state) {
  const int batch = static_cast<int>(state.range(0));
  const int q_heads = static_cast<int>(state.range(1));
  const int kv_heads = static_cast<int>(state.range(2));
  const int q_seq = static_cast<int>(state.range(3));
  const int kv_seq = static_cast<int>(state.range(4));

  const int group_size = q_heads / kv_heads;
  const int d_out = q_heads * HEAD_DIM;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  // cached_tokens in grouped_query_attention.cpp terminology
  const int cached_tokens = kv_seq - q_seq;

  GQABenchInputs<bfloat16> inputs(batch, q_heads, kv_heads, q_seq, kv_seq);

  nn::Softmax softmax;

  // Create mask like model.cpp does - size is max_tokens (here kv_seq)
  auto attn_mask = causal_attention_mask<int, CUDA>(kv_seq);

  // Slice like grouped_query_attention.cpp does for cache case:
  // slice(attn_mask, 0, cached_tokens, cached_tokens + input_seq_len)
  // slice(..., 1, 0, cached_tokens + input_seq_len)
  auto mask_to_use = slice(attn_mask.view(), 0, cached_tokens, cached_tokens + q_seq);
  mask_to_use = slice(mask_to_use.view(), 1, 0, cached_tokens + q_seq);
  auto attention_mask = mask_to_use.view();

  for (int i = 0; i < 20; ++i) {
    auto out = gqa_forward(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        attention_mask,
        softmax,
        scale,
        group_size,
        d_out);

    benchmark::DoNotOptimize(out.data());
  }
  cuda_check(cudaDeviceSynchronize(), "non-fused decode warmup sync");

  for (auto _ : state) {
    auto out = gqa_forward(
        inputs.q.view(),
        inputs.k.view(),
        inputs.v.view(),
        attention_mask,
        softmax,
        scale,
        group_size,
        d_out);

    benchmark::DoNotOptimize(out.data());
    cuda_check(cudaDeviceSynchronize(), "non-fused decode benchmark sync");
  }

  set_common_counters<bfloat16>(
      state, batch, q_heads, kv_heads, q_seq, kv_seq);
}

}  // namespace

BENCHMARK(BM_Llama_GQA_Core_Fused_Prefill)
    // batch, q_heads, kv_heads, seq_len
    // ->Args({1, 32, 8, 1024})
    // ->Args({1, 32, 8, 2048})
    // ->Args({1, 32, 8, 4096})
    // ->Args({2, 32, 8, 2048})
    ->Args({2, 32, 8, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Llama_GQA_Core_NonFused_Prefill)
    // batch, q_heads, kv_heads, seq_len
    // ->Args({1, 32, 8, 1024})
    // ->Args({1, 32, 8, 2048})
    // ->Args({1, 32, 8, 4096})
    // ->Args({2, 32, 8, 2048})
    ->Args({2, 32, 8, 4096})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// BENCHMARK(BM_Llama_GQA_Core_Fused_Decode)
//     // batch, q_heads, kv_heads, q_seq, kv_seq
//     ->Args({1,   32, 8, 1, 2048})
//     ->Args({16,  32, 8, 1, 2048})
//     ->Args({64,  32, 8, 1, 2048})
//     ->Args({128, 32, 8, 1, 2048})
//     ->Args({256, 32, 8, 1, 2048})
//     ->Args({128, 32, 8, 1, 4096})
//     ->Args({128, 32, 8, 1, 8192})
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();

// BENCHMARK(BM_Llama_GQA_Core_NonFused_Decode)
//     // batch, q_heads, kv_heads, q_seq, kv_seq
//     ->Args({1,   32, 8, 1, 2048})
//     ->Args({16,  32, 8, 1, 2048})
//     ->Args({64,  32, 8, 1, 2048})
//     ->Args({128, 32, 8, 1, 2048})
//     ->Args({256, 32, 8, 1, 2048})
//     ->Args({128, 32, 8, 1, 4096})
//     ->Args({128, 32, 8, 1, 8192})
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();

// BENCHMARK(BM_Llama_GQA_Core_Fused_Decode)
//     // chunked decode / small prefill
//     ->Args({16, 32, 8, 4, 2048})
//     ->Args({32, 32, 8, 4, 2048})
//     ->Args({64, 32, 8, 4, 2048})
//     ->Args({16, 32, 8, 8, 2048})
//     ->Args({32, 32, 8, 8, 2048})
//     ->Args({64, 32, 8, 8, 2048})
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();

// BENCHMARK(BM_Llama_GQA_Core_NonFused_Decode)
//     // chunked decode / small prefill
//     ->Args({16, 32, 8, 4, 2048})
//     ->Args({32, 32, 8, 4, 2048})
//     ->Args({64, 32, 8, 4, 2048})
//     ->Args({16, 32, 8, 8, 2048})
//     ->Args({32, 32, 8, 8, 2048})
//     ->Args({64, 32, 8, 8, 2048})
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();
