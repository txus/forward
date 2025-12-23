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

add_subdirectory(cpu)

if(APPLE)
  option(TENSOR_BUILD_CUDA "Build tensor_cuda library" OFF)
else()
  option(TENSOR_BUILD_CUDA "Build tensor_cuda library" ON)
endif()

if(TENSOR_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(cuda)
  target_compile_definitions(tensor_core INTERFACE TENSOR_HAS_CUDA)
endif()

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

