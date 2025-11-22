#include <cstdlib>
#include <forward/loader.hpp>
#include <iostream>
#include <print>
#include <tensor/tensor.hpp>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include <safetensors.hh>

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
std::string to_string(safetensors::dtype dtype, const uint8_t* data) {
  switch (dtype) {
  case safetensors::dtype::kBOOL: {
    return std::to_string((data[0] != 0U) ? 1 : 0);
  }
  case safetensors::dtype::kUINT8: {
    return std::to_string(data[0]);
  }
  case safetensors::dtype::kINT8: {
    return std::to_string(*reinterpret_cast<const int8_t*>(data));
  }
  case safetensors::dtype::kUINT16: {
    return std::to_string(*reinterpret_cast<const uint16_t*>(data));
  }
  case safetensors::dtype::kINT16: {
    return std::to_string(*reinterpret_cast<const int16_t*>(data));
  }
  case safetensors::dtype::kUINT32: {
    return std::to_string(*reinterpret_cast<const uint32_t*>(data));
  }
  case safetensors::dtype::kINT32: {
    return std::to_string(*reinterpret_cast<const int32_t*>(data));
  }
  case safetensors::dtype::kUINT64: {
    return std::to_string(*reinterpret_cast<const uint64_t*>(data));
  }
  case safetensors::dtype::kINT64: {
    return std::to_string(*reinterpret_cast<const int64_t*>(data));
  }
  case safetensors::dtype::kFLOAT16: {
    return std::to_string(safetensors::fp16_to_float(*reinterpret_cast<const uint16_t*>(data)));
  }
  case safetensors::dtype::kBFLOAT16: {
    return std::to_string(safetensors::bfloat16_to_float(*reinterpret_cast<const int64_t*>(data)));
  }
  case safetensors::dtype::kFLOAT32: {
    return std::to_string(*reinterpret_cast<const float*>(data));
  }
  case safetensors::dtype::kFLOAT64: {
    return std::to_string(*reinterpret_cast<const double*>(data));
  }
  }

  return {"???"};
}
// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

std::string to_string_snipped(const safetensors::tensor_t& tensor, const uint8_t* databuffer,
                              size_t count = 8) {
  std::stringstream string_stream;
  size_t nitems = safetensors::get_shape_size(tensor);
  size_t itembytes = safetensors::get_dtype_bytes(tensor.dtype);

  if ((count == 0) || ((count * 2) >= nitems)) {
    string_stream << "[";
    for (size_t i = 0; i < nitems; i++) {
      if (i > 0) {
        string_stream << ", ";
      }
      string_stream << to_string(tensor.dtype,
                                 databuffer + tensor.data_offsets[0] + (i * itembytes));
    }
    string_stream << "]";
  } else {
    string_stream << "[";
    size_t head_end = (std::min)(count, nitems);
    size_t tail_start = (std::max)(nitems - count, head_end);

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        string_stream << ", ";
      }
      string_stream << to_string(tensor.dtype,
                                 databuffer + tensor.data_offsets[0] + (i * itembytes));
    }

    string_stream << ", ..., ";

    for (size_t i = tail_start; i < nitems; i++) {
      if (i > tail_start) {
        string_stream << ", ";
      }
      string_stream << to_string(tensor.dtype,
                                 databuffer + tensor.data_offsets[0] + (i * itembytes));
    }

    string_stream << "]";
  }

  return string_stream.str();
}

safetensors::safetensors_t load(std::string_view(file_path)) {
  safetensors::safetensors_t safetensors;

  std::string warn;
  std::string err;
  bool ret = safetensors::mmap_from_file(std::string(file_path), &safetensors, &warn, &err);

  if (warn.size() != 0U) {
    std::println(std::cerr, "WARN: {}", warn);
  }

  if (!ret) {
    std::println(std::cerr, "Failed to load: {}\nError: {}", file_path, err);
    throw "fatal error";
  }

  if (!safetensors::validate_data_offsets(safetensors, err)) {
    std::println(std::cerr, "Invalid data offsets\nErr: {}", err);
    throw "fatal error";
  }

  return safetensors;
}

namespace loader {

std::vector<std::string> all_tensor_names(std::string_view file_path) {
  std::vector<std::string> out;

  safetensors::safetensors_t safetensors = load(file_path);

  out.reserve(safetensors.tensors.size());
  for (size_t i = 0; i < safetensors.tensors.size(); i++) {
    out.push_back(safetensors.tensors.keys()[i]);
  }

  return out;
}

void inspect_safetensors(std::string_view file_path) {
  safetensors::safetensors_t safetensors = load(file_path);

  const uint8_t* databuffer{safetensors.databuffer_addr};

  safetensors::tensor_t tensor;

  for (size_t i = 0; i < safetensors.tensors.size(); i++) {
    std::string key = safetensors.tensors.keys()[i];
    safetensors::tensor_t tensor;
    safetensors.tensors.at(i, &tensor);

    std::cout << key << ": " << safetensors::get_dtype_str(tensor.dtype) << " ";
    std::cout << "[";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << std::to_string(tensor.shape[i]);
    }
    std::cout << "]\n";

    std::cout << "  data_offsets[" << std::to_string(tensor.data_offsets[0]) << ", "
              << std::to_string(tensor.data_offsets[1]) << "]\n";
    std::cout << "  " << to_string_snipped(tensor, databuffer) << "\n";

    // Print metadata
    if (safetensors.metadata.size() != 0U) {
      std::cout << "\n";
      std::cout << "__metadata__\n";
      for (size_t i = 0; i < safetensors.metadata.size(); i++) {
        std::string key = safetensors.metadata.keys()[i];
        std::string value;
        safetensors.metadata.at(i, &value);

        std::cout << "  " << key << ":" << value << "\n";
      }
    }
  }
}

tensor::Tensor<tensor::bfloat16, tensor::CPU> load_from_safetensors(std::string_view file_path,
                                                                    std::string_view tensor_name) {
  safetensors::safetensors_t safetensors = load(file_path);

  safetensors::tensor_t tensor;
  bool res = safetensors.tensors.at(std::string(tensor_name), &tensor);
  if (!res) {
    throw std::runtime_error("Tensor not found");
  }

  if (tensor.dtype != safetensors::dtype::kBFLOAT16) {
    throw std::runtime_error("Only bf16 supported");
  }

  size_t nitems = safetensors::get_shape_size(tensor);

  std::vector<tensor::bfloat16> data(nitems);

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto* const bf16_data = reinterpret_cast<const tensor::bfloat16*>(
      safetensors.databuffer_addr + tensor.data_offsets[0]);
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  std::memcpy(data.data(), safetensors.databuffer_addr + tensor.data_offsets[0],
              nitems * sizeof(tensor::bfloat16));

  tensor::Tensor<tensor::bfloat16, tensor::CPU> out(tensor.shape, std::move(data));

  return out;
}
} // namespace loader
