file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

add_library(tensor_cuda STATIC storage.cu loader.cu ops.cu kernels/fill.cu kernels/arange.cu kernels/add.cu kernels/sub.cu kernels/div.cu kernels/mul.cu kernels/sum.cu kernels/max.cu)

set_target_properties(tensor_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Target CUDA architectures
#   70 = V100
#   75 = RTX 20xx, T4
#   80 = A100
#   86 = RTX 30xx
#   89 = RTX 40xx
#   90 = H100
#   100 = B200
#   120 = 5090 RTX
# Note: CUDA_ARCHITECTURES doesn't support generator expressions
# Use CMAKE_BUILD_TYPE to control this at configure time
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET tensor_cuda PROPERTY CUDA_ARCHITECTURES 120)
else()
  # RelWithDebInfo and Release: include PTX for future architectures
  set_property(TARGET tensor_cuda PROPERTY CUDA_ARCHITECTURES 120-real 120-virtual)
endif()

# Link against tensor_core and CUDA libraries
target_link_libraries(tensor_cuda
  PUBLIC
    tensor_core
    CUDA::cudart
    CUDA::cublas
  PRIVATE
    safetensors_cpp
)

target_include_directories(tensor_cuda
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
# CUDA compile options
target_compile_options(tensor_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    # Debug builds: Full debug info
    $<$<CONFIG:Debug>:
      -G              # Generate device debug info (disables optimizations)
      -g              # Generate host debug info
    >

    # RelWithDebInfo: Debug info + optimizations
    $<$<CONFIG:RelWithDebInfo>:
      -G              # Device debug info
      -g              # Host debug info
      --use_fast_math # Fast math even in debug
    >

    # Release: Maximum optimization
    $<$<CONFIG:Release>:
      -lineinfo
      --use_fast_math
    >

    # Common flags for all builds
    --expt-relaxed-constexpr
    -Xcompiler=-fPIC
  >
)
elseif (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang") # just for clangd
  # Use cuda-merged package which has complete headers, not just nvcc
  # Also specify resource-dir for NixOS where clangd uses a different resource directory
  # Include cuda_compat.h first to work around __noinline__ macro conflict
  set(CLANG_CUDA_FLAGS
    -fPIC
    -include ${CMAKE_CURRENT_SOURCE_DIR}/cuda_compat.h
  )
  if(DEFINED ENV{CUDA_PATH})
    list(APPEND CLANG_CUDA_FLAGS --cuda-path=$ENV{CUDA_PATH})
  endif()
  if(DEFINED CLANG_RESOURCE_DIR AND NOT CLANG_RESOURCE_DIR STREQUAL "")
    list(APPEND CLANG_CUDA_FLAGS -resource-dir=${CLANG_RESOURCE_DIR})
  endif()
  # NixOS clang wrapper injects GCC C++ includes which conflict with libc++
  # Use -nostdinc++ to disable auto-injection, then explicitly add libc++ headers
  # _ALLOW_UNSUPPORTED_LIBCPP bypasses CUDA's "libc++ not supported on x86" error
  if(DEFINED LIBCXX_INCLUDE AND NOT LIBCXX_INCLUDE STREQUAL "")
    list(APPEND CLANG_CUDA_FLAGS -nostdinc++ -cxx-isystem${LIBCXX_INCLUDE} -D_ALLOW_UNSUPPORTED_LIBCPP)
  endif()
  target_compile_options(tensor_cuda PRIVATE ${CLANG_CUDA_FLAGS})
endif()

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
