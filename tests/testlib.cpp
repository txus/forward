#include <gtest/gtest.h>

#include <forward/lib.hpp>

TEST(HelloTest, BasicAssertions) {
  int a = 5, b = 7;
  auto result = add_two(a, b);

  EXPECT_EQ(result, 12);
}
