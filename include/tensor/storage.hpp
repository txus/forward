#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <tensor/device.hpp>
#include <tensor/dtype.hpp>
#include <vector>

namespace tensor {

using namespace dtype;
using namespace device;

// Note: Using typename instead of DType/Device concepts to avoid ABI mismatch
template <typename T, typename D> class TensorStorage;

// Mutable CPU storage - owns or borrows mutable data
template <typename T> class TensorStorage<T, CPU> {
private:
  std::shared_ptr<T[]> data_; // NOLINT
  unsigned int size_ = 0;

public:
  using pointer = T*;
  using const_pointer = const T*;

  TensorStorage() = default;

  // Owning storage - allocates memory
  explicit TensorStorage(size_t size) : data_(new T[size]), size_(size) {}

  // Owning storage from vector (copies into shared_ptr)
  explicit TensorStorage(std::vector<T>&& vec) : data_(new T[vec.size()]), size_(vec.size()) {
    std::copy(vec.begin(), vec.end(), data_.get());
  }

  // Non-owning storage - borrows external mutable memory
  static TensorStorage borrow(T* ptr, size_t size) {
    TensorStorage storage;
    storage.data_ = std::shared_ptr<T[]>(ptr, [](T*) {}); // no-op deleter // NOLINT
    storage.size_ = size;
    return storage;
  }

  [[nodiscard]] size_t size() const {
    return size_;
  }
  pointer data() {
    return data_.get();
  }
  const_pointer data() const {
    return data_.get();
  }

  void resize(size_t size) {
    data_ = std::shared_ptr<T[]>(new T[size]); // NOLINT
    size_ = size;
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

// Const CPU storage - borrows read-only data (e.g., mmap)
template <typename T> class TensorStorage<const T, CPU> {
private:
  std::shared_ptr<const T[]> data_; // NOLINT
  size_t size_ = 0;

public:
  using pointer = const T*;
  using const_pointer = const T*;

  TensorStorage() = default;

  // Non-owning storage - borrows external read-only memory (e.g., mmap)
  static TensorStorage borrow(const T* ptr, size_t size) {
    TensorStorage storage;
    storage.data_ = std::shared_ptr<const T[]>(ptr, [](const T*) {}); // no-op deleter // NOLINT
    storage.size_ = size;
    return storage;
  }

  [[nodiscard]] size_t size() const {
    return size_;
  }
  const_pointer data() const {
    return data_.get();
  }

  const T& operator[](size_t idx) const {
    return data_[idx];
  }
};

#ifdef TENSOR_HAS_CUDA
// Mutable CUDA storage - owns device memory
template <typename T> class TensorStorage<T, CUDA> {
private:
  T* data_ = nullptr;
  size_t size_ = 0;

public:
  using pointer = T*;
  using const_pointer = const T*;

  TensorStorage() = default;
  explicit TensorStorage(size_t size);
  ~TensorStorage();

  // no copy, move only
  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;
  TensorStorage(TensorStorage&& other) noexcept;
  TensorStorage& operator=(TensorStorage&& other) noexcept;

  [[nodiscard]] size_t size() const {
    return size_;
  }
  pointer data() {
    return data_;
  }
  const_pointer data() const {
    return data_;
  }

  T operator[](size_t idx);
  const T operator[](size_t idx) const;

  void resize(size_t size);
  void fill(T value);
};

// Const CUDA storage - owns device memory but semantically read-only (e.g., loaded weights)
// Still needs to allocate and free device memory, just won't be mutated after loading
template <typename T> class TensorStorage<const T, CUDA> {
private:
  T* data_ = nullptr;
  size_t size_ = 0;

public:
  using pointer = const T*;
  using const_pointer = const T*;

  TensorStorage() = default;
  explicit TensorStorage(size_t size);
  ~TensorStorage();

  // no copy, move only
  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;
  TensorStorage(TensorStorage&& other) noexcept;
  TensorStorage& operator=(TensorStorage&& other) noexcept;

  [[nodiscard]] size_t size() const {
    return size_;
  }
  const_pointer data() const {
    return data_;
  }

  // Non-const data access for initial loading only
  T* mutable_data() {
    return data_;
  }

  T operator[](size_t idx);
  const T operator[](size_t idx) const;

  void resize(size_t size);
};
#endif

}; // namespace tensor
