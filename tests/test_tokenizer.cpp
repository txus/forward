#include <gtest/gtest.h>

#include <string>

#include <forward/tokenizer.hpp>

TEST(TokenizerTest, Roundtrip) {
  std::string prompt = "hello world";

  auto input_ids = encode("hello world");
  auto decoded = decode(input_ids);

  EXPECT_EQ(decoded, prompt);
}
