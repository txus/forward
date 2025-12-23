# testing library
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/common/test_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/common/test_config.h
)

# compiled library code
add_subdirectory(tensor/cpu)
add_subdirectory(nn)
add_subdirectory(llama)
add_subdirectory(forward)
