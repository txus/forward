#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <tensor/ops.hpp>

using namespace tensor;
using namespace benchmark;

static void BM_CUDA_AddBf16(State& state) {
  Tensor<bfloat16, CUDA> tensor_a(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});
  Tensor<bfloat16, CUDA> tensor_b(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  for (auto _ : state)
    DoNotOptimize(add(a_v, b_v));

  int64_t flops = 0;

  flops += state.iterations() * (state.range(0) * state.range(1));
  state.counters["FLOPs"] = Counter(flops, Counter::kIsRate);
  auto bytes_per_element = 2;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element * 2);
}

BENCHMARK(BM_CUDA_AddBf16)
    ->Args({8192, 2048})
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Args({262144, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CUDA_SumFp32LastDim(State& state) {
  Tensor<float, CUDA> tensor(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor.fill_(float(1.0));

  auto view = tensor.view();

  for (auto _ : state)
    DoNotOptimize(sum(view, -1, true));

  int64_t flops = 0;

  flops += state.iterations() * state.range(0) * state.range(1);
  state.counters["FLOPs"] = Counter(flops, Counter::kIsRate);
  auto bytes_per_element = 4;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element);
}

BENCHMARK(BM_CUDA_SumFp32LastDim)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CUDA_SumFp32FirstDim(State& state) {
  Tensor<float, CUDA> tensor(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor.fill_(float(1.0));

  auto view = tensor.view();

  for (auto _ : state)
    DoNotOptimize(sum(view, 0, true));

  int64_t flops = 0;

  flops += state.iterations() * state.range(0) * state.range(1);
  state.counters["FLOPs"] = Counter(flops, Counter::kIsRate);
  auto bytes_per_element = 4;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element);
}

BENCHMARK(BM_CUDA_SumFp32FirstDim)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CUDA_MaskedFillBf16(State& state) {
  Tensor<bfloat16, CUDA> tensor(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor.fill_(bfloat16(1.0));

  Tensor<int, CUDA> mask({static_cast<size_t>(state.range(1))});

  mask.fill_(1);

  auto view = tensor.view();

  for (auto _ : state)
    DoNotOptimize(masked_fill(view, mask.view(), 0.0));

  int64_t flops = 0;

  flops += state.iterations() * state.range(0) * state.range(1);
  state.counters["FLOPs"] = Counter(flops, Counter::kIsRate);
  auto bytes_per_element = 2;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element);
}

BENCHMARK(BM_CUDA_MaskedFillBf16)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();
