#pragma once

#include <stdexcept>
#include <tensor/tensor.hpp>
#include <unordered_map>

namespace loader {

void inspect_safetensors(std::string_view file_path);
std::vector<std::string> all_tensor_names(std::string_view file_path);
tensor::Tensor<tensor::bfloat16, tensor::CPU> load_from_safetensors(std::string_view file_path,
                                                                    std::string_view tensor_name);

template <tensor::DType T, tensor::Device D, typename... Names>
  requires(std::conjunction_v<std::is_convertible<Names, std::string_view>...>)
std::unordered_map<std::string, tensor::Tensor<T, D>> load_weights(std::string_view file_path,
                                                                   Names&&... names) { // NOLINT

  std::vector<std::string> name_list{std::string(std::string_view{names})...};

  std::unordered_map<std::string, tensor::Tensor<T, D>> out;

  for (auto& name : name_list) {
    try {
      auto tensor = load_from_safetensors(file_path, name);
      out.emplace(name, std::move(tensor));
    } catch (std::runtime_error) {
      fmt::println("Error loading tensor {}", name);
      throw;
    }
  }

  return out;
};

} // namespace loader
