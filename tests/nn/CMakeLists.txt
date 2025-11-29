add_executable(test_nn cpu/test_embedding.cpp cpu/test_rms_norm.cpp)
target_compile_features(test_nn PRIVATE cxx_std_26)
target_link_libraries(test_nn PRIVATE nn_core nn_cpu GTest::gtest_main)
target_include_directories(test_nn
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(GoogleTest)
gtest_discover_tests(test_nn)
