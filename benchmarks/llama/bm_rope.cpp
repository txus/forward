#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <llama/rope.hpp>
#include <tensor/device_type.hpp>

using namespace llama;
using namespace tensor;
using namespace benchmark;

const int batch_size = 4;
const int num_heads = 32;
const int seq_len = 65536;
const int head_dim = 64;

static void BM_Llama_RoPE(State& state) {
  ModelConfig conf{
      .head_dim = static_cast<size_t>(state.range(3)), .rope_theta = 500000.0, .max_position_embeddings = static_cast<size_t>(state.range(2))};

  RoPE<bfloat16, CUDA> rope{conf};

  Tensor<bfloat16, CUDA> inputs(
      {
        static_cast<size_t>(state.range(0)),
        static_cast<size_t>(state.range(1)),
        static_cast<size_t>(state.range(2)),
        static_cast<size_t>(state.range(3)),
    });

  inputs.fill_(bfloat16(4.0));

  auto view = inputs.view();

  for (auto _ : state)
    DoNotOptimize(rope.forward(inputs.view()));

  int64_t flops = 0;
 
  flops += state.iterations() * (state.range(0) * state.range(1) * state.range(2) * state.range(3)) + (state.range(3)/2);
  state.counters["FLOPs"] = Counter(flops, Counter::kIsRate);
  auto bytes_per_element = 2;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * state.range(2) * state.range(3) * bytes_per_element * 2);
}

BENCHMARK(BM_Llama_RoPE)
    ->Args({batch_size, num_heads, seq_len, head_dim})
    ->Unit(kMillisecond)
    ->UseRealTime();