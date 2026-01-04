include(CUDAConfig)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

add_library(tensor_cuda STATIC storage.cu loader.cu ops.cu kernels/fill.cu kernels/argmax.cu kernels/arange.cu kernels/sum.cu kernels/max.cu kernels/masked_fill.cu kernels/cat.cu kernels/map.cu kernels/zip.cu kernels/tril.cu kernels/slice.cu kernels/matmul.cu kernels/copy.cu)

configure_cuda_target(tensor_cuda)

target_link_libraries(tensor_cuda
  PUBLIC
    tensor_core
    CUDA::cudart
    CUDA::cublas
    CUDA::nvToolsExt
  PRIVATE
    safetensors_cpp
)

target_include_directories(tensor_cuda
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
