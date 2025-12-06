#pragma once

#include <fmt/color.h>
#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <span>
#include <stdexcept>
#include <tensor/device.hpp>
#include <tensor/dtype.hpp>
#include <utility>
#include <vector>

namespace tensor {

using namespace device;
using namespace dtype;

using Shape = std::vector<size_t>;

inline size_t get_stride(const Shape& shape, size_t dim) {
  auto dims_to_skip = dim + 1;
  size_t stride_ = 1;

  for (const auto& dim : shape) {
    if (dims_to_skip == 0) {
      stride_ *= dim;
    } else {
      dims_to_skip -= 1;
    }
  }

  return stride_;
}

inline Shape get_all_strides(const Shape& shape) {
  Shape strides{};
  for (size_t i = 0; i < shape.size(); ++i) {
    strides.push_back(get_stride(shape, i));
  }
  return strides;
}

template <DType T, Device D> class Tensor;

template <DType T, Device D> struct TensorView {
  std::span<T> data{};
  Shape shape;
  Shape stride;

  template <typename... Ix>
    requires(std::conjunction_v<std::is_integral<Ix>...>)
  TensorView get(Ix... dims) const {
    size_t indices[] = {static_cast<size_t>(dims)...}; // NOLINT
    size_t ndim = sizeof...(Ix);

    assert(ndim <= shape.size());

    size_t offset = 0;
    for (size_t idx = 0; idx < ndim; ++idx) {
      assert(indices[idx] < shape[idx]);
      size_t stride_ = stride[idx];
      offset += indices[idx] * stride_;
    }

    Shape new_shape;
    for (size_t dim = ndim; dim < shape.size(); ++dim) {
      new_shape.push_back(shape[dim]);
    }

    Shape new_strides;
    for (size_t dim = ndim; dim < stride.size(); ++dim) {
      new_strides.push_back(stride[dim]);
    }

    size_t sub_size = 1;
    for (auto dim : new_shape) {
      sub_size *= dim;
    }
    if (new_shape.empty()) {
      sub_size = 1; // scalar
    }

    return TensorView{std::span<T>(data.data() + offset, sub_size), new_shape, new_strides};
  }

  void transpose() {
    assert(shape.size() == 2);
    Shape new_shape = {shape[1], shape[0]};
    shape = new_shape;

    Shape new_strides = {stride[1], stride[0]};
    stride = new_strides;
  }

  Tensor<T, D> copy() const {
    Tensor<T, D> tensor{shape};

    assert(tensor.size() == data.size());
    std::copy_n(data.data(), data.size(), tensor.data());
    return tensor;
  }

  template <DType OutT> Tensor<OutT, D> map(std::function<OutT(T)> func) const {
    Tensor<OutT, D> result{shape};

    std::transform(data.begin(), data.end(), result.span().begin(), func);
    return result;
  }

  Tensor<float, D> to_float() const {
    return map<float>([](T val) { return static_cast<float>(val); });
  }

  TensorView<T, D> view_as(Shape new_shape) const {
    size_t total_elems = 1;
    for (size_t dim : new_shape) {
      total_elems *= dim;
    }
    assert(total_elems == data.size());

    TensorView<T, D> new_view;
    new_view.shape = std::move(new_shape);
    new_view.stride = get_all_strides(new_shape);
    new_view.data = data;
    return new_view;
  }

  Tensor<T, D> operator/(float other) const {
    return map<T>([other](T val) { return val / other; });
  }

  T item() const {
    assert(data.size() == 1);
    return data[0];
  }

  std::span<T> span() {
    return data;
  }
  [[nodiscard]] std::span<const T> span() const {
    return data;
  }
};

template <DType T, Device D> class Tensor {
private:
  std::vector<T> data_{};
  Shape shape_;

public:
  explicit Tensor(Shape shape) : shape_(std::move(shape)) {
    size_t total = 1;
    for (auto& dim : shape_) {
      total *= dim;
    }
    data_.resize(total);
  };
  explicit Tensor(Shape shape, std::vector<T>&& data)
      : shape_(std::move(shape)), data_(std::move(data)) {};
  ~Tensor() = default;

  TensorView<T, D> view() {
    return TensorView<T, D>{span(), shape(), get_all_strides(shape())};
  }
  TensorView<const T, D> view() const {
    return TensorView<const T, D>{span(), shape(), get_all_strides(shape())};
  }

  void fill_(T value) {
    std::fill(data_.begin(), data_.end(), value);
  }

  void set_(int idx, T value) {
    if (idx >= size()) {
      fmt::print("Error setting {} at idx {} on a tensor sized {}", value, idx, size());
      throw std::out_of_range("cannot set beyond size");
    }

    span()[idx] = value;
  }

  T item() const {
    assert(shape().size() == 0);
    return data_.data()[0];
  }

  [[nodiscard]] Shape shape() const {
    return shape_;
  }

  std::span<T> span() {
    return {data(), size()};
  }

  [[nodiscard]] std::span<const T> span() const {
    return {data(), size()};
  }

  T* data() {
    return data_.data();
  }
  const T* data() const {
    return data_.data();
  }

  [[nodiscard]] size_t size() const {
    return data_.size();
  }

  T at(int idx) const {
    if (idx > size()) {
      throw std::out_of_range("cannot index past the tensor size");
    }
    return data_[idx];
  }
};

} // namespace tensor

template <tensor::DType T, tensor::Device D> struct fmt::formatter<tensor::TensorView<T, D>> {
  // no custom format spec for now -> just {}
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin(); // no format options
  }

  template <typename FormatContext>
  auto format(const tensor::TensorView<T, D> tensor_view, FormatContext& ctx) const {
    auto out = ctx.out();

    fmt::format_to(out, "<Tensor<{}, {}> ", tensor::dtype::dtype_name<T>::value,
                   tensor::device::device_name<D>::value);
    fmt::format_to(out, fmt::emphasis::italic, "shape=[");

    for (std::size_t i = 0; i < tensor_view.shape.size(); ++i) {
      if (i > 0) {
        out = fmt::format_to(out, ", ");
      }
      out = fmt::format_to(out, fmt::fg(fmt::color::aqua), "{}", tensor_view.shape[i]);
    }

    out = fmt::format_to(out, fmt::emphasis::italic, "] ");

    out = format_tensor_view(out, tensor_view);

    out = fmt::format_to(out, fmt::emphasis::bold, ">");

    return out;
  }

private:
  template <typename OutputIt>
  OutputIt format_tensor_view(OutputIt out, const tensor::TensorView<T, D> tensor_view) const {
    constexpr std::size_t max_elems_per_dim = 4; // tweak as you like
    return format_tensor_rec(out, tensor_view, /*dim=*/0, /*offset=*/0, max_elems_per_dim);
  }

  template <typename OutputIt>
  OutputIt format_tensor_rec(OutputIt out, const tensor::TensorView<T, D> tensor_view,
                             std::size_t dim, std::size_t offset, std::size_t max_elems) const {
    const auto& shape = tensor_view.shape;
    const auto& strides = tensor_view.stride;
    if (dim == shape.size()) {
      // Base case: actually print one scalar
      return fmt::format_to(out, "{}", tensor_view.span()[offset]);
    }

    auto dim_size = shape[dim];
    auto stride = strides[dim];

    *out++ = '[';
    size_t count = std::min<size_t>(dim_size, max_elems);
    for (size_t i = 0; i < count; ++i) {
      if (i > 0) {
        out = fmt::format_to(out, ", ");
      }
      out = format_tensor_rec(out, tensor_view, dim + 1, offset + (i * stride), max_elems);
    }
    if (dim_size > max_elems) {
      out = fmt::format_to(out, ", ...");
    }
    *out++ = ']';
    return out;
  }
};
