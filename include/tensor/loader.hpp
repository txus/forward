#pragma once

#include <tensor/tensor.hpp>

namespace safetensors {
class safetensors_t;
} // namespace safetensors

namespace tensor {

template <DType T, Device D> class Loader {
private:
  std::unique_ptr<safetensors::safetensors_t> safetensors_;

public:
  explicit Loader(std::string_view file_path);
  ~Loader();

  void inspect() const;
  tensor::TensorView<const T, D> load(std::string_view tensor_name, bool transpose = false) const;
};

} // namespace tensor
