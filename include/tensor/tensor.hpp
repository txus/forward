#pragma once

#include <algorithm>
#include <cassert>
#include <fmt/color.h>
#include <fmt/format.h>
#include <iostream>
#include <print>
#include <span>
#include <stdexcept>
#include <vector>

namespace tensor {

typedef std::vector<size_t> Shape;

inline const size_t stride(Shape shape, size_t dim) {
  auto dims_to_skip = dim + 1;
  auto stride_ = 1;

  for (auto &dim : shape) {
    if (dims_to_skip == 0) {
      stride_ *= dim;
    } else {
      dims_to_skip -= 1;
    }
  }

  return stride_;
}

template <typename T> class Tensor;

template <typename T> struct TensorView {
  std::span<T> data;
  Shape shape;

  template <typename... Ix>
    requires(std::conjunction_v<std::is_integral<Ix>...>)
  TensorView get(Ix... dims) const {
    size_t indices[] = {static_cast<size_t>(dims)...};
    size_t ndim = sizeof...(Ix);

    assert(ndim <= shape.size());

    size_t offset = 0;
    for (size_t d = 0; d < ndim; ++d) {
      assert(indices[d] < shape[d]);
      size_t s = stride(shape, d);
      offset += indices[d] * s;
    }

    Shape new_shape;
    for (size_t d = ndim; d < shape.size(); ++d) {
      new_shape.push_back(shape[d]);
    }

    size_t sub_size = 1;
    for (auto d : new_shape)
      sub_size *= d;
    if (new_shape.empty())
      sub_size = 1; // scalar

    return TensorView{std::span<T>(data.data() + offset, sub_size), new_shape};
  }

  Tensor<T> copy() const {
    Tensor<T> t{shape};

    assert(t.size() == data.size());
    std::copy_n(data.data(), data.size(), t.data());
    return t;
  }

  T item() const {
    assert(data.size() == 1);
    return data[0];
  }

  std::span<T> span() { return data; }
  std::span<const T> span() const { return data; }
};

template <typename T> class Tensor {
private:
  std::vector<T> data_;
  const Shape shape_;

public:
  explicit Tensor(Shape shape) : shape_(shape) {
    size_t total = 1;
    for (auto dim : shape)
      total *= dim;
    data_.resize(total);
  };
  explicit Tensor(Shape shape, std::vector<T> &&data)
      : shape_(shape), data_(std::move(data)) {};
  ~Tensor() = default;

  TensorView<T> view() { return TensorView<T>{span(), shape()}; }
  TensorView<T> view() const { return TensorView<T>{span(), shape()}; }

  void fill_(T value) { std::fill(data_.begin(), data_.end(), value); }

  void set_(int idx, T value) {
    if (idx >= size()) {
      fmt::print("Error setting {} at idx {} on a tensor sized {}", value, idx,
                 size());
      throw std::out_of_range("cannot set beyond size");
    }

    data_[idx] = value;
  }

  T item() const {
    assert(shape().size() == 0);
    return data_.data()[0];
  }

  const Shape shape() const { return shape_; }

  std::span<T> span() { return {data(), size()}; }

  std::span<const T> span() const { return {data(), size()}; }

  T *data() { return data_.data(); }
  const T *data() const { return data_.data(); }

  size_t size() const { return data_.size(); }

  T at(int idx) const {
    if (idx > size()) {
      throw std::out_of_range("cannot index past the tensor size");
    }
    return data_[idx];
  }
};

template <typename T> bool all_close(T &&a, T &&b, float eps) {
  if (a->size() != b->size())
    throw std::invalid_argument(
        "cannot compare two tensors of different shapes");

  for (int i = 0; i < a->size(); ++i) {
    if (std::abs(a->raw()[i] - b->raw()[i]) > eps) {
      return false;
    }
  }
  return true;
}

template <typename T>
void assert_all_close(T &&a, T &&b, float eps,
                      const char *err_msg = "assert_all_close failed") {
  if (!(all_close(a, b, eps)))
    throw std::runtime_error(err_msg);
}

} // namespace tensor

template <typename T> struct fmt::formatter<tensor::TensorView<T>> {
  // no custom format spec for now -> just {}
  constexpr auto parse(format_parse_context &ctx) {
    return ctx.begin(); // no format options
  }

  template <typename FormatContext>
  auto format(const tensor::TensorView<T> &tv, FormatContext &ctx) const {
    auto out = ctx.out();

    fmt::format_to(out, fmt::emphasis::bold, "<TensorView ");
    fmt::format_to(out, fmt::emphasis::italic, "shape=[");

    for (std::size_t i = 0; i < tv.shape.size(); ++i) {
      if (i > 0)
        out = fmt::format_to(out, ", ");
      out = fmt::format_to(out, fmt::fg(fmt::color::aqua), "{}", tv.shape[i]);
    }

    out = fmt::format_to(out, fmt::emphasis::italic, "] ");

    out = format_tensor_view(out, tv);

    out = fmt::format_to(out, fmt::emphasis::bold, ">");

    return out;
  }

private:
  template <typename OutputIt>
  OutputIt format_tensor_view(OutputIt out,
                              const tensor::TensorView<T> &tv) const {
    constexpr std::size_t max_elems_per_dim = 4; // tweak as you like
    return format_tensor_rec(out, tv, /*dim=*/0, /*offset=*/0,
                             max_elems_per_dim);
  }

  template <typename OutputIt>
  OutputIt format_tensor_rec(OutputIt out, const tensor::TensorView<T> &tv,
                             std::size_t dim, std::size_t offset,
                             std::size_t max_elems) const {
    const auto &shape = tv.shape;
    if (dim == shape.size()) {
      // Base case: actually print one scalar
      return fmt::format_to(out, "{}", tv.span()[offset]);
    }

    auto dim_size = shape[dim];
    auto stride = tensor::stride(shape, dim); // assuming you have this

    *out++ = '[';
    std::size_t n = std::min<std::size_t>(dim_size, max_elems);
    for (std::size_t i = 0; i < n; ++i) {
      if (i > 0) {
        out = fmt::format_to(out, ", ");
      }
      out = format_tensor_rec(out, tv, dim + 1, offset + i * stride, max_elems);
    }
    if (dim_size > max_elems) {
      out = fmt::format_to(out, ", ...");
    }
    *out++ = ']';
    return out;
  }
};
