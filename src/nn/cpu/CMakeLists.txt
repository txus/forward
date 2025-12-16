file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/nn/*.hpp")

add_library(nn_cpu STATIC embedding.cpp act.cpp rms_norm.cpp linear.cpp softmax.cpp)

target_link_libraries(nn_cpu
  PUBLIC tensor_core tensor_cpu nn_core
)

target_include_directories(nn_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

