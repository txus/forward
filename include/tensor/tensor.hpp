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
#include <tensor/storage.hpp>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef TENSOR_HAS_CUDA
#include <cuda_runtime.h>
#endif

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

inline Shape broadcast_shape(const Shape& shape_a, const Shape& shape_b) {
  assert(shape_a.size() == shape_b.size());

  Shape out;
  for (size_t i = 0; i < shape_a.size(); ++i) {
    assert(shape_a[i] == shape_b[i] || shape_a[i] == 1 || shape_b[i] == 1);
    out.push_back(std::max(shape_a[i], shape_b[i]));
  }
  return out;
}

inline Shape broadcast_strides(const Shape& shape, const Shape& strides, const Shape& out_shape) {
  Shape result;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1 && out_shape[i] > 1) {
      result.push_back(0); // broadcast: don't advance
    } else {
      result.push_back(strides[i]);
    }
  }
  return result;
}

inline std::vector<size_t> linear_to_multidim(size_t linear_idx, Shape shape) {
  std::vector<size_t> index;

  for (size_t dim = 0; dim < shape.size(); ++dim) {
    size_t stride = 1;
    for (size_t dim_ = dim + 1; dim_ < shape.size(); ++dim_) {
      stride *= shape[dim_];
    }

    index.push_back(linear_idx / stride % shape[dim]);
  }

  return index;
}

template <DType T, Device D> class Tensor;

template <DType T, Device D> struct TensorView {
  T* data = nullptr;
  size_t data_size = 0;

  Shape shape;
  Shape stride;

  // Default constructor
  TensorView() = default;

  // Main constructor
  TensorView(T* data_, size_t size_, Shape shape_, Shape stride_)
      : data(data_), data_size(size_), shape(std::move(shape_)), stride(std::move(stride_)) {}

  // Converting constructor: allow TensorView<T, D> to convert to TensorView<const T, D>
  template <typename U>
    requires std::same_as<U, std::remove_const_t<T>> && std::is_const_v<T>
  TensorView(const TensorView<U, D>& other)
      : data(other.data), data_size(other.data_size), shape(other.shape), stride(other.stride) {}

  std::span<T> span()
    requires std::same_as<D, device::CPU>
  {
    return std::span<T>(data, data_size);
  }

  std::span<const T> span() const
    requires std::same_as<D, device::CPU>
  {
    return std::span<const T>(data, data_size);
  }

  [[nodiscard]] size_t total_elements() const {
    size_t out = 1;
    for (auto dim : shape) {
      out *= dim;
    }
    return out;
  }

  template <typename... Ix>
    requires(std::conjunction_v<std::is_integral<Ix>...>)
  TensorView<T, D> get(Ix... dims) {
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

    return TensorView{data + offset, sub_size, new_shape, new_strides};
  }

  template <typename... Ix>
    requires(std::conjunction_v<std::is_integral<Ix>...>)
  TensorView<const T, D> get(Ix... dims) const {
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

    return TensorView{data + offset, sub_size, new_shape, new_strides};
  }

  void transpose(size_t dim_a, size_t dim_b) {
    assert(shape.size() >= std::max(dim_a, dim_b) + 1);

    Shape new_shape{};
    Shape new_stride{};
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (dim == dim_a) {
        new_shape.push_back(shape[dim_b]);
        new_stride.push_back(stride[dim_b]);
      } else if (dim == dim_b) {
        new_shape.push_back(shape[dim_a]);
        new_stride.push_back(stride[dim_a]);
      } else {
        new_shape.push_back(shape[dim]);
        new_stride.push_back(stride[dim]);
      }
    }

    shape = new_shape;
    stride = new_stride;
  }

  void transpose() {
    assert(shape.size() == 2);
    transpose(0, 1);
  }

  Tensor<std::remove_const_t<T>, D> repeat_interleave(size_t dim, size_t repeats) const {
    assert(dim < shape.size());

    Shape temp_shape;
    Shape temp_stride;

    for (size_t dim_ = 0; dim_ <= dim; ++dim_) {
      temp_shape.push_back(shape[dim_]);
      temp_stride.push_back(stride[dim_]);
    }

    temp_shape.push_back(repeats);
    temp_stride.push_back(0);

    for (size_t dim_ = dim + 1; dim_ < shape.size(); ++dim_) {
      temp_shape.push_back(shape[dim_]);
      temp_stride.push_back(stride[dim_]);
    }

    size_t temp_size = 1;
    for (auto dim_ : temp_shape) {
      temp_size *= dim_;
    }

    TensorView temp_view{data, temp_size, temp_shape, temp_stride};

    Tensor<T, D> materialized = temp_view.copy();

    Shape final_shape;
    for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
      if (dim_ == dim) {
        final_shape.push_back(shape[dim_] * repeats); // Expanded dimension
      } else {
        final_shape.push_back(shape[dim_]);
      }
    }

    return materialized.view().reshape(final_shape);
  }

  [[nodiscard]] bool is_contiguous() const {
    if (shape.empty()) {
      return true;
    }

    size_t expected_stride = 1;
    for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
      if (stride[dim] != expected_stride) {
        return false;
      }
      expected_stride *= shape[dim];
    }
    return true;
  }

  template <DType OutT, typename Func> Tensor<OutT, D> map(Func func) const {
    Tensor<OutT, D> result{shape};

    auto result_span = result.span();

    size_t total_elems = total_elements();

    for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
      std::vector<size_t> indices = linear_to_multidim(linear_idx, shape);

      size_t offset = 0;
      for (size_t dim = 0; dim < shape.size(); ++dim) {
        offset += indices[dim] * stride[dim];
      }

      result_span[linear_idx] = func(data[offset]);
    }

    return result;
  }

  template <typename Func> void each(Func func) const {
    size_t total_elems = total_elements();

    for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
      std::vector<size_t> indices = linear_to_multidim(linear_idx, shape);

      size_t offset = 0;
      for (size_t dim = 0; dim < shape.size(); ++dim) {
        offset += indices[dim] * stride[dim];
      }

      func(data[offset]);
    }
  }

  template <DType OutT> Tensor<OutT, D> to() const {
    return map<OutT>([](T val) { return static_cast<OutT>(val); });
  }

  void check_for_nans() const {
    for (size_t i = 0; i < span().size(); ++i) {
      if (std::isnan(span()[i])) {
        fmt::println("Tensor has NaN at index {}: {}", i, span()[i]);
        break;
      }
      if (std::isinf(span()[i])) {
        fmt::println("Tensor has Inf at index {}: {}", i, span()[i]);
        break;
      }
    }
  }

  Tensor<std::remove_const_t<T>, D> copy() const {
    return map<std::remove_const_t<T>>([](T val) { return val; });
  }

  Tensor<std::remove_const_t<T>, D> contiguous() const {
    Tensor<std::remove_const_t<T>, D> result{shape};
    auto dst_span = result.span();

    std::function<void(size_t, std::vector<size_t>&)> iterate;
    iterate = [&](size_t dim, std::vector<size_t>& indices) {
      if (dim == shape.size()) {
        // Compute source offset using actual strides
        size_t src_offset = 0;
        for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
          src_offset += indices[dim_] * stride[dim_];
        }

        // Compute destination offset assuming row-major (contiguous) layout
        size_t dst_offset = 0;
        for (size_t dim_ = 0; dim_ < shape.size(); ++dim_) {
          size_t dst_stride = 1;
          for (size_t dd = dim_ + 1; dd < shape.size(); ++dd) {
            dst_stride *= shape[dd];
          }
          dst_offset += indices[dim_] * dst_stride;
        }

        dst_span[dst_offset] = data[src_offset];
        return;
      }

      for (size_t i = 0; i < shape[dim]; ++i) {
        indices[dim] = i;
        iterate(dim + 1, indices);
      }
    };

    std::vector<size_t> indices(shape.size());
    iterate(0, indices);

    return result;
  }

  Tensor<std::remove_const_t<T>, D> reshape(Shape new_shape) const {
    size_t total_elems = 1;
    for (size_t dim : new_shape) {
      total_elems *= dim;
    }
    assert(total_elems == data_size);

    auto out = Tensor<std::remove_const_t<T>, D>{new_shape};

    // TODO: check that this is correct

    replace_from_(out, *this);

    return out;
  }

  Tensor<std::remove_const_t<T>, D> cos() const {
    return map<std::remove_const_t<T>>([](T val) { return std::cos(val); });
  }

  Tensor<std::remove_const_t<T>, D> sin() const {
    return map<std::remove_const_t<T>>([](T val) { return std::sin(val); });
  }

  Tensor<std::remove_const_t<T>, D> exp() const {
    return map<std::remove_const_t<T>>([](T val) { return std::exp(val); });
  }

  T item() const {
    assert(data_size == 1);
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
  TensorStorage<T, D> storage_;
  Shape shape_;

public:
  explicit Tensor() : shape_({0}) {}

  // Allocating constructor - only for mutable tensors
  explicit Tensor(Shape shape)
    requires(!std::is_const_v<T>)
      : shape_(std::move(shape)) {
    size_t total = 1;
    for (auto& dim : shape_) {
      total *= dim;
    }
    storage_.resize(total);
  };

  // Storage-taking constructor - works for both const and non-const
  explicit Tensor(Shape shape, TensorStorage<T, D>&& storage)
      : shape_(std::move(shape)), storage_(std::move(storage)) {}

  // Vector constructor - only for mutable CPU tensors
  explicit Tensor(Shape shape, std::vector<std::remove_const_t<T>>&& vec)
    requires std::same_as<D, device::CPU> && (!std::is_const_v<T>)
      : shape_(std::move(shape)), storage_(std::move(vec)) {}

  ~Tensor() = default;

  // Move constructor and assignment
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  // Delete copy (storage may be non-copyable)
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  typename TensorStorage<T, D>::pointer data() {
    return storage_.data();
  }
  typename TensorStorage<T, D>::const_pointer data() const {
    return storage_.data();
  }

  [[nodiscard]] size_t size() const {
    return storage_.size();
  }
  [[nodiscard]] Shape shape() const {
    return shape_;
  }

  TensorView<T, D> view() {
    return TensorView<T, D>{data(), size(), shape(), get_all_strides(shape())};
  }

  TensorView<const T, D> view() const {
    return TensorView<const T, D>{data(), size(), shape(), get_all_strides(shape())};
  }

  // Copy to a new mutable tensor
  Tensor<std::remove_const_t<T>, D> copy() const {
    return view().copy();
  }

  void fill_(T value)
    requires(!std::is_const_v<T>)
  {
    storage_.fill(value);
  }

#ifdef TENSOR_HAS_CUDA
  // Device transfer methods

  Tensor<std::remove_const_t<T>, CUDA> cuda() const
    requires std::same_as<D, device::CPU>
  {
    Tensor<std::remove_const_t<T>, CUDA> result{shape()};
    cudaMemcpy(result.data(), data(), size() * sizeof(T), cudaMemcpyHostToDevice);
    return result;
  }

  Tensor<std::remove_const_t<T>, CPU> cpu() const
    requires std::same_as<D, device::CUDA>
  {
    Tensor<std::remove_const_t<T>, CPU> result{shape()};
    cudaMemcpy(result.data(), data(), size() * sizeof(T), cudaMemcpyDeviceToHost);
    return result;
  }
#endif

  // CPU specific useful things

  std::span<T> span()
    requires std::same_as<D, device::CPU>
  {
    return std::span<T>(data(), size());
  }

  std::span<const T> span() const
    requires std::same_as<D, device::CPU>
  {
    return std::span<const T>(data(), size());
  }

  void set_(int idx, T value)
    requires std::same_as<D, device::CPU>
  {
    if (idx >= size()) {
      fmt::print("Error setting {} at idx {} on a tensor sized {}", value, idx, size());
      throw std::out_of_range("cannot set beyond size");
    }

    span()[idx] = value;
  }

  T item() const {
    assert(shape().size() == 0);
    return storage_.data()[0];
  }

  T at(int idx) const {
    if (idx > size()) {
      throw std::out_of_range("cannot index past the tensor size");
    }
    return storage_[idx];
  }
};

} // namespace tensor

template <> struct fmt::formatter<tensor::Shape> {
  constexpr auto static parse(format_parse_context& ctx) {
    return ctx.begin(); // no format options
  }

  template <typename FormatContext> auto format(tensor::Shape shape, FormatContext& ctx) const {
    auto out = ctx.out();

    fmt::format_to(out, fmt::emphasis::italic, "[");

    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        out = fmt::format_to(out, ", ");
      }
      out = fmt::format_to(out, fmt::fg(fmt::color::aqua), "{}", shape[i]);
    }
    fmt::format_to(out, fmt::emphasis::italic, "]");

    return out;
  }
};

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
    fmt::format_to(out, "shape={}", tensor_view.shape);

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
