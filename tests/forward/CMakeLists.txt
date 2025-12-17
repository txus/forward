add_executable(test_forward test_tokenizer.cpp)
target_compile_features(test_forward PRIVATE cxx_std_26)
target_link_libraries(test_forward PRIVATE forward_core GTest::gtest_main)
target_include_directories(test_forward
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(GoogleTest)
gtest_discover_tests(test_forward)
