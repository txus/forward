#pragma once

#include <cmath>
#include <gtest/gtest.h>
#include <span>
#include <tensor/tensor.hpp>

template <typename T>
void tensor_is_close(std::span<const T> a, std::span<const T> b,
                     float atol = float(1e-5), float rtol = float(1e-5)) {
  ASSERT_EQ(a.size(), b.size()) << "Span sizes differ";

  for (size_t i = 0; i < a.size(); ++i) {
    T diff = std::fabs(a[i] - b[i]);
    T limit = atol + rtol * std::fabs(b[i]);
    ASSERT_LE(diff, limit) << "Mismatch at index " << i << ": a=" << a[i]
                           << " b=" << b[i] << " diff=" << diff
                           << " limit=" << limit;
  }
}
