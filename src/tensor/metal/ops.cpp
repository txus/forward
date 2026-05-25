#include <tensor/ops.hpp>

template <>
Tensor<bfloat16, CUDA> fill(const TensorView<bfloat16, CUDA>& input, bfloat16 masked_value) {}
