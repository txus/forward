#add_executable(test_llama test_layer.cpp test_rope.cpp test_grouped_query_attention.cpp test_mlp.cpp)
add_executable(test_llama test_grouped_query_attention.cpp)

target_compile_features(test_llama PRIVATE cxx_std_26)
target_link_libraries(test_llama PRIVATE llama GTest::gtest_main)
target_include_directories(test_llama
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(GoogleTest)
gtest_discover_tests(test_llama)
