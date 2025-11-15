#include "test_config.h"
#include <gtest/gtest.h>

#include <forward/model.hpp>

TEST(ModelTest, Forward) {

  try {
    model::Model mod(TEST_MODEL_PATH);

    // auto input = tensor::Tensor<int>{{1, 4}};
    // auto output = mod.forward(input);
  } catch (std::exception e) {
    std::println("Exception we're cooked {}", e.what());
  }

  // EXPECT_EQ(completion, "completion");
}
