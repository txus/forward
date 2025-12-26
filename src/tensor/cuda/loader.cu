#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

#include "../common/utils.h"
#include "kernels/utils.cuh"

namespace tensor {

using namespace dtype;
using namespace device;

template <typename T, typename D>
Loader<T, D>::Loader(std::string_view file_path) : safetensors_(load_safetensors(file_path)){};
template <typename T, typename D> Loader<T, D>::~Loader() = default;

template <typename T, typename D> void Loader<T, D>::inspect() const {
  inspect_safetensors(safetensors_);
}

// CUDA loader: copies from mmap'd host memory to device memory
template <typename T, typename D>
[[nodiscard]] Tensor<const T, D> Loader<T, D>::load(std::string_view tensor_name) const {
  fmt::println("Loading {} to CUDA...", tensor_name);

  safetensors::tensor_t tensor;
  bool res = safetensors_->tensors.at(std::string(tensor_name), &tensor);
  if (!res) {
    throw std::runtime_error("Tensor not found");
  }

  if (tensor.dtype != safetensors::dtype::kBFLOAT16) {
    throw std::runtime_error("Only bf16 supported");
  }

  size_t nitems = safetensors::get_shape_size(tensor);

  const auto* host_data = reinterpret_cast<const T*>(          // NOLINT
      safetensors_->databuffer_addr + tensor.data_offsets[0]); // NOLINT

  Shape shape = tensor.shape;

  // Allocate device memory
  TensorStorage<const T, D> storage(static_cast<int>(nitems));

  // Copy from host (mmap) to device
  CUDA_CHECK(cudaMemcpy(storage.mutable_data(), host_data, nitems * sizeof(T), cudaMemcpyHostToDevice));

  return Tensor<const T, D>{shape, std::move(storage)};
}

template class Loader<bfloat16, CUDA>;

} // namespace tensor
