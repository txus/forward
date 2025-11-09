#include "test_config.h"
#include <gtest/gtest.h>

#include <forward/tokenizer.hpp>

TEST(TokenizerTest, Roundtrip) {
  tokenizer::Tokenizer tok(TEST_MODEL_PATH "/tokenizer.json");
  const std::string_view prompt = "hello world";

  auto input_ids = tok.encode(prompt);
  auto decoded = tok.decode(input_ids);

  EXPECT_EQ(decoded, prompt);
}
