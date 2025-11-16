#include <cstdlib>
#include <forward/loader.hpp>
#include <iostream>
#include <print>
#include <tensor/tensor.hpp>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include <safetensors.hh>

std::string to_string(safetensors::dtype dtype, const uint8_t *data) {
  switch (dtype) {
  case safetensors::dtype::kBOOL: {
    return std::to_string(data[0] ? 1 : 0);
  }
  case safetensors::dtype::kUINT8: {
    return std::to_string(data[0]);
  }
  case safetensors::dtype::kINT8: {
    return std::to_string(*reinterpret_cast<const int8_t *>(data));
  }
  case safetensors::dtype::kUINT16: {
    return std::to_string(*reinterpret_cast<const uint16_t *>(data));
  }
  case safetensors::dtype::kINT16: {
    return std::to_string(*reinterpret_cast<const int16_t *>(data));
  }
  case safetensors::dtype::kUINT32: {
    return std::to_string(*reinterpret_cast<const uint32_t *>(data));
  }
  case safetensors::dtype::kINT32: {
    return std::to_string(*reinterpret_cast<const int32_t *>(data));
  }
  case safetensors::dtype::kUINT64: {
    return std::to_string(*reinterpret_cast<const uint64_t *>(data));
  }
  case safetensors::dtype::kINT64: {
    return std::to_string(*reinterpret_cast<const int64_t *>(data));
  }
  case safetensors::dtype::kFLOAT16: {
    return std::to_string(
        safetensors::fp16_to_float(*reinterpret_cast<const uint16_t *>(data)));
  }
  case safetensors::dtype::kBFLOAT16: {
    return std::to_string(safetensors::bfloat16_to_float(
        *reinterpret_cast<const int64_t *>(data)));
  }
  case safetensors::dtype::kFLOAT32: {
    return std::to_string(*reinterpret_cast<const float *>(data));
  }
  case safetensors::dtype::kFLOAT64: {
    return std::to_string(*reinterpret_cast<const double *>(data));
  }
  }

  return std::string("???");
}

std::string to_string_snipped(const safetensors::tensor_t &t,
                              const uint8_t *databuffer, size_t N = 8) {
  std::stringstream ss;
  size_t nitems = safetensors::get_shape_size(t);
  size_t itembytes = safetensors::get_dtype_bytes(t.dtype);

  if ((N == 0) || ((N * 2) >= nitems)) {
    ss << "[";
    for (size_t i = 0; i < nitems; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }
    ss << "]";
  } else {
    ss << "[";
    size_t head_end = (std::min)(N, nitems);
    size_t tail_start = (std::max)(nitems - N, head_end);

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }

    ss << ", ..., ";

    for (size_t i = tail_start; i < nitems; i++) {
      if (i > tail_start) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }

    ss << "]";
  }

  return ss.str();
}

safetensors::safetensors_t load(std::string_view(file_path)) {
  safetensors::safetensors_t st;

  std::string warn, err;
  bool ret =
      safetensors::mmap_from_file(std::string(file_path), &st, &warn, &err);

  if (warn.size()) {
    std::println(std::cerr, "WARN: {}", warn);
  }

  if (!ret) {
    std::println(std::cerr, "Failed to load: {}\nError: {}", file_path, err);
    throw "fatal error";
  }

  if (!safetensors::validate_data_offsets(st, err)) {
    std::println(std::cerr, "Invalid data offsets\nErr: {}", err);
    throw "fatal error";
  }

  return st;
}

namespace loader {

std::vector<std::string> all_tensor_names(std::string_view file_path) {
  std::vector<std::string> out;

  safetensors::safetensors_t st = load(file_path);

  for (size_t i = 0; i < st.tensors.size(); i++) {
    out.push_back(st.tensors.keys()[i]);
  }

  return out;
}

void inspect_safetensors(std::string_view file_path) {
  safetensors::safetensors_t st = load(file_path);

  const uint8_t *databuffer{st.databuffer_addr};

  safetensors::tensor_t tensor;

  for (size_t i = 0; i < st.tensors.size(); i++) {
    std::string key = st.tensors.keys()[i];
    safetensors::tensor_t tensor;
    st.tensors.at(i, &tensor);

    std::cout << key << ": " << safetensors::get_dtype_str(tensor.dtype) << " ";
    std::cout << "[";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << std::to_string(tensor.shape[i]);
    }
    std::cout << "]\n";

    std::cout << "  data_offsets[" << std::to_string(tensor.data_offsets[0])
              << ", " << std::to_string(tensor.data_offsets[1]) << "]\n";
    std::cout << "  " << to_string_snipped(tensor, databuffer) << "\n";

    // Print metadata
    if (st.metadata.size()) {
      std::cout << "\n";
      std::cout << "__metadata__\n";
      for (size_t i = 0; i < st.metadata.size(); i++) {
        std::string key = st.metadata.keys()[i];
        std::string value;
        st.metadata.at(i, &value);

        std::cout << "  " << key << ":" << value << "\n";
      }
    }
  }
}

tensor::Tensor<float> load_from_safetensors(std::string_view file_path,
                                            std::string_view tensor_name) {
  safetensors::safetensors_t st = load(file_path);

  safetensors::tensor_t tensor;
  bool res = st.tensors.at(std::string(tensor_name), &tensor);
  if (!res) {
    throw std::runtime_error("Tensor not found");
  }

  if (tensor.dtype != safetensors::dtype::kBFLOAT16) {
    throw std::runtime_error("Only bf16 supported");
  }

  size_t nitems = safetensors::get_shape_size(tensor);

  std::vector<float> data(nitems);

  const uint16_t *bf16_data = reinterpret_cast<const uint16_t *>(
      st.databuffer_addr + tensor.data_offsets[0]);

  std::transform(bf16_data, bf16_data + nitems, data.begin(),
                 safetensors::bfloat16_to_float);

  tensor::Tensor<float> out(tensor.shape, std::move(data));

  return out;
}
} // namespace loader
