#include <fmt/format.h>
#include <gtest/gtest.h>

#include <forward/tokenizer.hpp>

#include "common/test_config.h"

TEST(ForwardTokenizerTest, Roundtrip) {
  tokenizer::Tokenizer tok(TEST_MODEL_PATH "/tokenizer.json");
  const std::string_view prompt = "hello world";

  std::vector<int> input_ids = tok.encode(prompt);
  auto decoded = tok.decode(input_ids);

  EXPECT_EQ(decoded, fmt::format("<|begin_of_text|>{}", prompt));
}
