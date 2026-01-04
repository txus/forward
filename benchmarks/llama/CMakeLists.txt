add_executable(bm_llama bm_rope.cpp)
target_link_libraries(bm_llama PRIVATE llama benchmark::benchmark_main)

target_compile_features(bm_llama PRIVATE cxx_std_26)

target_include_directories(bm_llama
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
