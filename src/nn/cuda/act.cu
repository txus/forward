#include <nn/act.hpp>
#include "kernels/act.cuh"

using namespace nn;
using namespace tensor;

namespace nn {

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> SiLU::operator()(const TensorView<T, D>& input) const {
  return kernels::silu(input);
}

template Tensor<bfloat16, CUDA> SiLU::operator()(const TensorView<bfloat16, CUDA>& input) const;

}
