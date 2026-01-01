# Common CUDA configuration for all CUDA targets
#
# Usage:
#   include(CUDAConfig)
#   configure_cuda_target(my_cuda_target)

# Target CUDA architectures
#   70 = V100
#   75 = RTX 20xx, T4
#   80 = A100
#   86 = RTX 30xx
#   89 = RTX 40xx
#   90 = H100
#   100 = B200
#   120 = 5090 RTX

function(configure_cuda_target TARGET_NAME)
  set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Note: CUDA_ARCHITECTURES doesn't support generator expressions
  # Use CMAKE_BUILD_TYPE to control this at configure time
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 120)
  else()
    # RelWithDebInfo and Release: include PTX for future architectures
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 120-real 120-virtual)
  endif()

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # CUDA compile options for nvcc
    target_compile_options(${TARGET_NAME} PRIVATE
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
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # Clang CUDA flags (for clangd compatibility)
    # Use cuda-merged package which has complete headers, not just nvcc
    # Also specify resource-dir for NixOS where clangd uses a different resource directory
    set(CLANG_CUDA_FLAGS -fPIC)

    # Include cuda_compat.h if it exists in the source directory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_compat.h")
      list(APPEND CLANG_CUDA_FLAGS -include ${CMAKE_CURRENT_SOURCE_DIR}/cuda_compat.h)
    endif()

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

    target_compile_options(${TARGET_NAME} PRIVATE ${CLANG_CUDA_FLAGS})
  endif()
endfunction()
