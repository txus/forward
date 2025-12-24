#pragma once

#include <memory>
#include <string_view>
#include <tensor/tensor.hpp>

namespace safetensors {
class safetensors_t;
} // namespace safetensors

namespace tensor {

// Note: Using typename instead of DType/Device concepts to avoid ABI mismatch
template <typename T, typename D> class Loader {
private:
  std::unique_ptr<safetensors::safetensors_t> safetensors_;

public:
  explicit Loader(std::string_view file_path);
  ~Loader();

  void inspect() const;
  Tensor<const T, D> load(std::string_view tensor_name) const;
};

}; // namespace tensor
