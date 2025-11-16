#include "common/test_config.h"
#include <gtest/gtest.h>

#include <forward/loader.hpp>

TEST(LoaderTest, LoadFromSafetensors) {
  auto path = std::string(TEST_MODEL_PATH "/model.safetensors");
  auto weights = loader::load_weights(path, "model.embed_tokens.weight");

  auto embed_weights = weights.at("model.embed_tokens.weight");

  tensor::Shape shape{{128256, 2048}};

  EXPECT_EQ(embed_weights.shape(), shape);

  auto w = embed_weights.view().span();

  float expected_first = 0.004517;
  float expected_last = 0.006622;

  float epsilon = 0.0001;

  EXPECT_LT(std::abs(w[0] - expected_first), epsilon);
  EXPECT_LT(std::abs(w[w.size() - 1] - expected_last), epsilon);
}
