#pragma once

#include <forward/tensor.hpp>

namespace loader {
void inspect_safetensors(std::string_view file_path);
tensor::Tensor<float> load_from_safetensors(std::string_view file_path,
                                            std::string_view tensor_name);
} // namespace loader
