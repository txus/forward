#include "test_config.h"
#include <gtest/gtest.h>

#include <forward/loader.hpp>

TEST(LoaderTest, LoadFromSafetensors) {
  tensor::Tensor<float> t = loader::load_from_safetensors(
      TEST_MODEL_PATH "/model.safetensors", "model.embed_tokens.weight");

  tensor::Shape shape = {128256, 2048};

  EXPECT_EQ(t.shape(), shape);

  float expected_first = 0.004517;
  float expected_last = 0.006622;

  float epsilon = 0.0001;

  EXPECT_LT(std::abs(t.raw()[0] - expected_first), epsilon);
  EXPECT_LT(std::abs(t.raw()[t.size() - 1] - expected_last), epsilon);
}
