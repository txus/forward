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

if(APPLE)
  option(NN_BUILD_CUDA "Build nn_cuda library" OFF)
else()
  option(NN_BUILD_CUDA "Build nn_cuda library" ON)
endif()

if(NN_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(cuda)
  target_compile_definitions(nn_core INTERFACE NN_HAS_CUDA)
endif()

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

