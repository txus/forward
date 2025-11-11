#pragma once

#include <algorithm>
#include <exception>
#include <iostream>
#include <print>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace tensor {

typedef std::vector<size_t> Shape;

template <typename T> class Tensor {
private:
  std::vector<T> data_;
  Shape shape_;

public:
  explicit Tensor(Shape shape) {
    size_t total = 1;
    for (auto dim : shape)
      total *= dim;
    data_.resize(total);
    shape_ = shape;
  };
  explicit Tensor(Shape shape, std::vector<T> &&data)
      : shape_(shape), data_(std::move(data)) {};
  ~Tensor() = default;

  void fill_(T value) { std::fill(data_.begin(), data_.end(), value); }

  void set_(int idx, T value) {
    if (idx >= size()) {
      std::println(std::cerr, "Error setting {} at idx {} on a tensor sized {}",
                   value, idx, size());
      throw std::out_of_range("cannot set beyond size");
    }

    data_[idx] = value;
  }

  std::vector<T> raw() const { return data_; }
  Shape shape() const { return shape_; }

  size_t size() const { return data_.size(); }

  size_t stride(size_t dim) const {
    auto dims_to_skip = dim + 1;
    auto stride_ = 1;

    for (auto &dim : shape_) {
      if (dims_to_skip == 0) {
        stride_ *= dim;
      } else {
        dims_to_skip -= 1;
      }
    }

    return stride_;
  }

  T at(int idx) const {
    if (idx > size()) {
      throw std::out_of_range("cannot index past the tensor size");
    }
    return data_[idx];
  }

  ///
  /// Slices an N-rank tensor on the first dimension, returning an (N-1)-rank
  /// tensor.
  ///
  Tensor<T> slice(size_t dim0_idx) const { // copying slice, very inefficient
    if (shape_.size() < 1) {
      throw std::out_of_range("cannot slice into a rank-0 tensor");
    }
    if (dim0_idx >= shape_.at(0)) {
      throw std::out_of_range("dim0 index out of range");
    }

    Shape new_shape{};
    // start at dim 1, because we squeezed dim 0
    for (int i = 1; i < shape_.size(); ++i) {
      new_shape.push_back(shape_.at(i));
    }

    Tensor<T> out(new_shape);

    auto stride_ = stride(0);
    auto start = dim0_idx * stride_;

    for (int offset = start, i = 0; i < stride_; ++offset, ++i) {
      out.data_[i] = data_[offset];
    }

    return out;
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
