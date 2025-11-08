#include <forward/lib.hpp>

#include <fmt/format.h>

int main() {
  int a = 5, b = 7;

  auto result = add_two(a, b);

  fmt::print("Result: {}", result);

  return 0;
}
