#include <nn/linear.hpp>
#include <tensor/loader.hpp>
#include <tensor/ops.hpp>
#include <util/nvtx.hpp>

using namespace nn;
using namespace tensor;

template <typename T, typename D>
void Linear<T, D>::load_weights(const tensor::Loader<T, D>& loader, std::string_view name) {
  weights_ = loader.load(name);
  weights_t_ = weights_.view();
  weights_t_.transpose();
}

template <typename T, typename D>
Tensor<T, D> Linear<T, D>::forward(const TensorView<T, D>& inputs) {
  NVTX_RANGE("linear");
  // cuBLAS matmul detects that weights_t_ is a 2D transpose and uses CUBLAS_OP_T
  // to handle it efficiently without copying data.
  return matmul(inputs, weights_t_);
}

template class nn::Linear<bfloat16, CUDA>;
