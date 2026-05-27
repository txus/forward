#include <tensor/ops.hpp>

#include "launcher.hpp"

namespace tensor {

using namespace dtype;
using namespace device;
using namespace metal;

template <typename T, typename D> Tensor<T, D> arange(T start, T end, T step) {
  auto n_elements = static_cast<uint32_t>((end - start) / step);
  TensorStorage<T, D> storage(n_elements);
  Shape shape{n_elements};
  Tensor<T, D> out{shape, std::move(storage)};

  const char* fn_name = nullptr;
  if constexpr (std::is_same_v<T, bfloat16>) {
    fn_name = "arange_bf16";
  } else if constexpr (std::is_same_v<T, float>) {
    fn_name = "arange_f32";
  } else if constexpr (std::is_same_v<T, int>) {
    fn_name = "arange_i32";
  }

  metal::launch(fn_name, n_elements,
                {to_buffer(out), to_bytes(start), to_bytes(step), to_bytes(n_elements)});

  return out;
}

template Tensor<bfloat16, METAL> arange<bfloat16, METAL>(bfloat16 start, bfloat16 end,
                                                         bfloat16 step);
template Tensor<float, METAL> arange<float, METAL>(float start, float end, float step);
template Tensor<int, METAL> arange<int, METAL>(int start, int end, int step);

} // namespace tensor
