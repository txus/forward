if(BACKEND_CUDA)
add_executable(bm_tensor cpu/bm_ops.cpp cuda/bm_ops.cpp)
target_link_libraries(bm_tensor PRIVATE tensor_core tensor_cpu tensor_cuda benchmark::benchmark_main)
else()
add_executable(bm_tensor cpu/bm_ops.cpp)
target_link_libraries(bm_tensor PRIVATE tensor_core tensor_cpu benchmark::benchmark_main)
endif()

target_compile_features(bm_tensor PRIVATE cxx_std_26)

target_include_directories(bm_tensor
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
