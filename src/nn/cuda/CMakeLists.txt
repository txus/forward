include(CUDAConfig)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/nn/*.hpp")

add_library(nn_cuda STATIC act.cu embedding.cu linear.cu rms_norm.cu softmax.cu kernels/act.cu kernels/embedding.cu kernels/rms_norm.cu)

configure_cuda_target(nn_cuda)

target_link_libraries(nn_cuda
  PUBLIC
    tensor_core tensor_cuda nn_core
)

target_include_directories(nn_cuda
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
