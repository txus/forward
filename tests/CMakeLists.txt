# testing library
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Tests need to be added as executables firts
add_executable(testlib testlib.cpp)
target_compile_features(testlib PRIVATE cxx_std_26)
target_link_libraries(testlib PRIVATE forward GTest::gtest_main)

add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_features(test_tokenizer PRIVATE cxx_std_26)
target_link_libraries(test_tokenizer PRIVATE forward GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(testlib)
gtest_discover_tests(test_tokenizer)
