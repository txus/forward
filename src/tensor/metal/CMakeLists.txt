include(MetalShaders)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

add_library(tensor_metal STATIC metal_cpp.cpp metal_context.cpp storage.cpp ops.cpp)

target_link_libraries(tensor_metal
  PUBLIC
    tensor_core
  PRIVATE
    metal_cpp
    safetensors_cpp
)

compile_metal_shaders(tensor_metal
  kernels/fill.metal
  kernels/arange.metal
  )

target_include_directories(tensor_metal
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
