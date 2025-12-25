#include <gtest/gtest.h>

#include <common/test_utils.hpp>
#include <tensor/loader.hpp>

#include "common/test_config.h"

using namespace tensor;

TEST(TensorCUDALoaderTest, LoadFromSafetensors) {
  SKIP_IF_NO_GPU();
  Loader<bfloat16, CUDA> loader{TEST_WEIGHTS_PATH};

  auto embed_weights = loader.load("model.embed_tokens.weight");

  tensor::Shape shape{{128256, 2048}};

  EXPECT_EQ(embed_weights.shape(), shape);

  auto cpu_weights = embed_weights.cpu();
  auto weights_ = cpu_weights.span();

  fmt::println("Weights {}", cpu_weights.view());

  bfloat16 expected_first = 0.004517;
  bfloat16 expected_last = 0.006622;

  bfloat16 epsilon = 0.0001;

  EXPECT_LT(std::abs(weights_[0] - expected_first), epsilon);
  EXPECT_LT(std::abs(weights_[weights_.size() - 1] - expected_last), epsilon);
}
