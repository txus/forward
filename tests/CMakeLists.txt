# testing library
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Option to skip CUDA tests at runtime (for CI without GPU)
option(SKIP_CUDA_TESTS "Skip CUDA tests at runtime (use when GPU unavailable)" OFF)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/common/test_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/common/test_config.h
)

# compiled library code
if(APPLE)
  option(TENSOR_BUILD_CUDA "Build tensor_cuda library" OFF)
else()
  option(TENSOR_BUILD_CUDA "Build tensor_cuda library" ON)
endif()

if(TENSOR_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(tensor/cuda)
endif()

add_subdirectory(tensor/cpu)
add_subdirectory(nn)
add_subdirectory(llama)
add_subdirectory(forward)
