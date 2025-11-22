add_executable(test_tensor test_tensor.cpp cpu/test_ops.cpp)
target_compile_features(test_tensor PRIVATE cxx_std_26)
target_link_libraries(test_tensor PRIVATE tensor_core tensor_cpu GTest::gtest_main)
target_include_directories(test_tensor
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(GoogleTest)
gtest_discover_tests(test_tensor)
