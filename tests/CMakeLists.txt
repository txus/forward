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

# Im using C++26 in the test
target_compile_features(testlib PRIVATE cxx_std_26)

# should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(testlib PRIVATE forward GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(testlib)
