file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/nn/*.hpp")

add_library(nn_core INTERFACE)

target_include_directories(nn_core
  INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(nn_core
    INTERFACE
        cxx_std_26
)

target_sources(nn_core
    INTERFACE
        ${HEADER_LIST}
)

add_subdirectory(cpu)

if(BACKEND_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(cuda)
endif()

# Unified nn library - aggregates CPU and backend-specific libraries
add_library(nn INTERFACE)
target_link_libraries(nn INTERFACE nn_core nn_cpu)

if(BACKEND_CUDA)
  target_link_libraries(nn INTERFACE nn_cuda)
endif()

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

