#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tensor {

template <typename T> class Tensor {
private:
  std::vector<T> data_;
  std::vector<size_t> shape_;

public:
  explicit Tensor(std::vector<size_t> shape) {
    size_t total = 1;
    for (auto dim : shape)
      total *= dim;
    data_.resize(total);
    shape_ = shape;
  };
  ~Tensor() = default;

  void fill_(T value) { std::fill(data_.begin(), data_.end(), value); }

  std::vector<T> raw() const { return data_; }
  std::vector<size_t> shape() const { return shape_; }

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

  Tensor<T> slice(size_t dim0_idx) const { // copying slice, very inefficient
    if (shape_.size() < 1) {
      throw std::out_of_range("cannot slice into a rank-0 tensor");
    }
    if (dim0_idx >= shape_.at(0)) {
      throw std::out_of_range("dim0 index out of range");
    }

    auto dims_to_skip{1};
    std::vector<size_t> new_shape{};
    for (auto &dim : shape_) {
      if (dims_to_skip == 0) {
        new_shape.push_back(dim);
      }
      dims_to_skip -= 1;
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
} // namespace tensor
