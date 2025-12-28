add_executable(test_tensor_cpu test_tensor.cpp test_loader.cpp test_ops.cpp)
target_compile_features(test_tensor_cpu PRIVATE cxx_std_26)
target_link_libraries(test_tensor_cpu PRIVATE tensor_core tensor_cpu GTest::gtest_main)

target_include_directories(test_tensor_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

include(GoogleTest)
gtest_discover_tests(test_tensor_cpu)
