#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <tensor/ops.hpp>

using namespace tensor;
using namespace benchmark;

static void BM_CPU_AddBf16(State& state) {
  Tensor<bfloat16, CPU> tensor_a(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});
  Tensor<bfloat16, CPU> tensor_b(
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

BENCHMARK(BM_CPU_AddBf16)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CPU_SumFp32LastDim(State& state) {
  Tensor<float, CPU> tensor(
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

BENCHMARK(BM_CPU_SumFp32LastDim)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CPU_SumFp32FirstDim(State& state) {
  Tensor<float, CPU> tensor(
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

BENCHMARK(BM_CPU_SumFp32FirstDim)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CPU_MaskedFillBf16(State& state) {
  Tensor<bfloat16, CPU> tensor(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor.fill_(bfloat16(1.0));

  Tensor<int, CPU> mask({static_cast<size_t>(state.range(1))});

  mask.fill_(1);

  auto view = tensor.view();

  for (auto _ : state)
    DoNotOptimize(masked_fill(view, mask.view(), 0.0));

  state.counters["FLOPs"] = Counter(0, Counter::kIsRate);
  auto bytes_per_element = 2;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element);
}

BENCHMARK(BM_CPU_MaskedFillBf16)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CPU_CatBf16FirstDim(State& state) {
  Tensor<bfloat16, CPU> tensor_a(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});
  Tensor<bfloat16, CPU> tensor_b(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  for (auto _ : state) {
    DoNotOptimize(cat(a_v, b_v, 0));
  }

  state.counters["FLOPs"] = Counter(0, Counter::kIsRate);
  auto bytes_per_element = 2;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element * 2);
}

BENCHMARK(BM_CPU_CatBf16FirstDim)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();

static void BM_CPU_CatBf16LastDim(State& state) {
  Tensor<bfloat16, CPU> tensor_a(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});
  Tensor<bfloat16, CPU> tensor_b(
      {static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1))});

  tensor_a.fill_(bfloat16(4.0));
  tensor_b.fill_(bfloat16(3.0));

  auto a_v = tensor_a.view();
  auto b_v = tensor_b.view();

  for (auto _ : state) {
    DoNotOptimize(cat(a_v, b_v, 1));
  }

  state.counters["FLOPs"] = Counter(0, Counter::kIsRate);
  auto bytes_per_element = 2;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * state.range(0) *
                          state.range(1) * bytes_per_element * 2);
}

BENCHMARK(BM_CPU_CatBf16LastDim)
    ->Args({16384, 2048})
    ->Args({65536, 2048})
    ->Unit(kMillisecond)
    ->UseRealTime();
