#pragma once

#include <cstddef>
#include <tensor/device.hpp>
#include <tensor/dtype.hpp>
#include <vector>

namespace tensor {

using namespace dtype;
using namespace device;

template <DType T, Device D> class TensorStorage;

template <DType T> class TensorStorage<T, CPU> {
private:
  std::vector<T> data_;

public:
  using pointer = T*;
  using const_pointer = const T*;

  TensorStorage() = default;
  explicit TensorStorage(size_t size) : data_(size) {}
  explicit TensorStorage(std::vector<T>&& data) : data_(std::move(data)) {}

  size_t size() const {
    return data_.size();
  }
  pointer data() {
    return data_.data();
  }
  const_pointer data() const {
    return data_.data();
  }

  void resize(size_t size) {
    data_.resize(size);
  }
  void fill(T value) {
    std::fill(data(), data() + size(), value);
  }

  T& operator[](size_t idx) {
    return data_[idx];
  }
  const T& operator[](size_t idx) const {
    return data_[idx];
  }
};

template <DType T> class TensorStorage<T, CUDA> {
private:
  T* data_ = nullptr;
  int size_ = 0;

public:
  using pointer = T*;
  using const_pointer = const T*;

  TensorStorage() = default;
  explicit TensorStorage(int size);
  ~TensorStorage();

  // no copy, move only
  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;
  TensorStorage(TensorStorage&& other) noexcept;
  TensorStorage& operator=(TensorStorage&& other) noexcept;

  int size() const {
    return size_;
  }
  pointer data() {
    return data_;
  }
  const_pointer data() const {
    return data_;
  }

  void resize(int size);
  void fill(T value);
};
}; // namespace tensor
