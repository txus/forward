#include <gtest/gtest.h>

#include <forward/loader.hpp>

#include "common/test_config.h"

using namespace tensor;

TEST(LoaderTest, LoadFromSafetensors) {
  auto path = std::string(TEST_MODEL_PATH "/model.safetensors");
  auto weights = loader::load_weights<bfloat16, CPU>(path, "model.embed_tokens.weight");

  auto embed_weights = weights.at("model.embed_tokens.weight");

  tensor::Shape shape{{128256, 2048}};

  EXPECT_EQ(embed_weights.shape(), shape);

  auto weights_ = embed_weights.view().span();

  fmt::println("Weights {}", embed_weights.view());

  bfloat16 expected_first = 0.004517;
  bfloat16 expected_last = 0.006622;

  bfloat16 epsilon = 0.0001;

  EXPECT_LT(std::abs(weights_[0] - expected_first), epsilon);
  EXPECT_LT(std::abs(weights_[weights_.size() - 1] - expected_last), epsilon);
}
