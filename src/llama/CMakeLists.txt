file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/llama/*.hpp")

option(FUSED_ROPE "Use fused RoPE kernel (CUDA only)" ON)
option(FUSED_ATTN "Use fused attention kernel (CUDA only)" ON)

set(LLAMA_SOURCES config.cpp model.cpp layer.cpp mlp.cpp rope.cpp grouped_query_attention.cpp kv_cache.cpp)

if(BACKEND_CUDA)
  include(CUDAConfig)
  enable_language(CUDA)
  list(APPEND LLAMA_SOURCES kernels/rope.cu kernels/grouped_query_attention.cu)
elseif(FUSED_ATTN)
  message(WARNING "FUSED_ATTN requested but BACKEND_CUDA=OFF; ignoring")
elseif(FUSED_ROPE)
  message(WARNING "FUSED_ROPE requested but BACKEND_CUDA=OFF; ignoring")
endif()

add_library(llama STATIC ${LLAMA_SOURCES})

if(BACKEND_CUDA)
  configure_cuda_target(llama)
endif()

target_link_libraries(llama
  PUBLIC tensor nn
  PRIVATE nlohmann_json::nlohmann_json
)

target_include_directories(llama
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(BACKEND_CUDA)
  if(FUSED_ROPE)
    target_compile_definitions(llama PRIVATE FUSED_ROPE)
    message(STATUS "Fused RoPE: enabled")
  else()
    message(STATUS "Fused RoPE: disabled")
  endif()

  if(FUSED_ATTN)
    target_compile_definitions(llama PRIVATE FUSED_ATTN)
    message(STATUS "Fused GQA: enabled")
  else()
    message(STATUS "Fused GQA: disabled")
  endif()
endif()

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
