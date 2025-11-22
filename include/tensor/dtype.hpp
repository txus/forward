#pragma once

#include <fmt/format.h>

#include <bit>
#include <type_traits>

namespace tensor::dtype {
struct bfloat16 {
  uint16_t bits;

  bfloat16() = default;
  bfloat16(float value) { // extract top 16 bits
    auto f32_bits = std::bit_cast<uint32_t>(value);
    bits = static_cast<uint16_t>(f32_bits >> 16);
  }

  // pad with zeros at the bottom
  operator float() const {
    auto f32_bits = static_cast<uint32_t>(bits << 16);

    return std::bit_cast<float>(f32_bits);
  };

  bfloat16 operator+(bfloat16 other) const {
    return {float(*this) + float(other)};
  }

  bfloat16 operator-(bfloat16 other) const {
    return {float(*this) - float(other)};
  }

  bfloat16 operator*(bfloat16 other) const {
    return {float(*this) * float(other)};
  }

  bfloat16 operator/(bfloat16 other) const {
    return {float(*this) / float(other)};
  }
};

template <typename T> struct is_dtype : std::false_type {};

template <> struct is_dtype<bfloat16> : std::true_type {};
template <> struct is_dtype<int> : std::true_type {};

template <typename D>
concept DType = is_dtype<D>::value;

template <typename T> struct dtype_name;
template <> struct dtype_name<bfloat16> {
  static constexpr const char* value = "bfloat16";
};
template <> struct dtype_name<int> {
  static constexpr const char* value = "int";
};
} // namespace tensor::dtype

template <> struct fmt::formatter<tensor::dtype::bfloat16> {
  static constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const tensor::dtype::bfloat16& bf16, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "{}", float(bf16));
  }
};
