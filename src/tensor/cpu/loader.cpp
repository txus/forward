#include <cstdlib>
#include <tensor/loader.hpp>
#include <tensor/tensor.hpp>

#include "../common/utils.h"

namespace tensor {

template <DType T, Device D>
Loader<T, D>::Loader(std::string_view file_path) : safetensors_(load_safetensors(file_path)){};
template <DType T, Device D> Loader<T, D>::~Loader() = default;

template <DType T, Device D> void Loader<T, D>::inspect() const {
  inspect_safetensors(safetensors_);
}

// bfloat16 only for now
template <DType T, Device D>
[[nodiscard]] Tensor<const T, D> Loader<T, D>::load(std::string_view tensor_name) const {
  fmt::println("Loading {}...", tensor_name);

  safetensors::tensor_t tensor;
  bool res = safetensors_->tensors.at(std::string(tensor_name), &tensor);
  if (!res) {
    throw std::runtime_error("Tensor not found");
  }

  if (tensor.dtype != safetensors::dtype::kBFLOAT16) {
    throw std::runtime_error("Only bf16 supported");
  }

  size_t nitems = safetensors::get_shape_size(tensor);

  const auto* data = reinterpret_cast<const T*>(               // NOLINT
      safetensors_->databuffer_addr + tensor.data_offsets[0]); // NOLINT

  Shape shape = tensor.shape;
  auto storage = TensorStorage<const T, D>::borrow(data, nitems);

  return Tensor<const T, D>{shape, std::move(storage)};
}

template class Loader<bfloat16, CPU>;

} // namespace tensor
