add_executable(test_nn_cpu test_embedding.cpp test_rms_norm.cpp test_act.cpp test_softmax.cpp)
target_compile_features(test_nn_cpu PRIVATE cxx_std_26)
target_link_libraries(test_nn_cpu PRIVATE nn_core nn_cpu GTest::gtest_main)
target_include_directories(test_nn_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

include(GoogleTest)
gtest_discover_tests(test_nn_cpu)
