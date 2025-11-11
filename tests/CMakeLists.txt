# testing library
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_config.h
)

add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_features(test_tokenizer PRIVATE cxx_std_26)
target_link_libraries(test_tokenizer PRIVATE forward GTest::gtest_main)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_model test_model.cpp)
target_compile_features(test_model PRIVATE cxx_std_26)
target_link_libraries(test_model PRIVATE forward GTest::gtest_main)
target_include_directories(test_model PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_tensor test_tensor.cpp)
target_compile_features(test_tensor PRIVATE cxx_std_26)
target_link_libraries(test_tensor PRIVATE forward GTest::gtest_main)

add_executable(test_loader test_loader.cpp)
target_compile_features(test_loader PRIVATE cxx_std_26)
target_link_libraries(test_loader PRIVATE forward GTest::gtest_main)
target_include_directories(test_loader PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_embedding nn/test_embedding.cpp)
target_compile_features(test_embedding PRIVATE cxx_std_26)
target_link_libraries(test_embedding PRIVATE forward GTest::gtest_main)
target_include_directories(test_embedding PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

include(GoogleTest)
gtest_discover_tests(test_tokenizer)
gtest_discover_tests(test_model)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_loader)
gtest_discover_tests(test_embedding)
