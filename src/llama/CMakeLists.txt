file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/llama/*.hpp")

set(LLAMA_SOURCES config.cpp model.cpp layer.cpp mlp.cpp rope.cpp grouped_query_attention.cpp kv_cache.cpp)

if(BACKEND_CUDA)
  include(CUDAConfig)
  enable_language(CUDA)
  list(APPEND LLAMA_SOURCES rope.cu)
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

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
