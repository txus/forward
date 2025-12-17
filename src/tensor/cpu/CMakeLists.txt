file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

add_library(tensor_cpu STATIC ops.cpp loader.cpp)

target_link_libraries(tensor_cpu
  PUBLIC tensor_core
  PRIVATE safetensors_cpp
)

target_include_directories(tensor_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

