file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

add_library(tensor_core INTERFACE)

target_link_libraries(tensor_core
    INTERFACE
        fmt::fmt
)

target_include_directories(tensor_core
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(tensor_core
    INTERFACE
        cxx_std_26
)

target_sources(tensor_core
    INTERFACE
        ${HEADER_LIST}
)

# Find CUDA toolkit before processing subdirectories so includes are available
if(BACKEND_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(cpu)

if(BACKEND_CUDA)
  add_subdirectory(cuda)
endif()

# Unified tensor library - aggregates CPU and backend-specific libraries
add_library(tensor INTERFACE)
target_link_libraries(tensor INTERFACE tensor_core tensor_cpu)

if(BACKEND_CUDA)
  target_link_libraries(tensor INTERFACE tensor_cuda)
endif()

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

