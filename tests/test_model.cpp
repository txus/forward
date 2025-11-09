#include "test_config.h"
#include <gtest/gtest.h>

#include <forward/model.hpp>

TEST(ModelTest, Forward) {
  model::Model mod(TEST_MODEL_PATH);

  auto completion = mod.generate("Hello, world", 5);

  EXPECT_EQ(completion, "completion");
}
