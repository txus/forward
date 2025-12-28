# Disable the Google Benchmark requirement on Google Test
set(BENCHMARK_ENABLE_TESTING NO)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)

FetchContent_MakeAvailable(googlebenchmark)

if(APPLE)
  option(TENSOR_CUDA "Benchmark tensor_cuda library" OFF)
else()
  option(TENSOR_CUDA "Benchmark tensor_cuda library" ON)
endif()

add_subdirectory(tensor)
