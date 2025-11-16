file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/llama/*.hpp")

FetchContent_Declare(
  safetensors_cpp
  GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp.git
  GIT_TAG 10f7d8f)
set(SAFETENSORS_CPP_CXX_EXCEPTIONS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(safetensors_cpp)

add_library(llama STATIC embedding.cpp model.cpp)

target_link_libraries(llama
  PUBLIC tensor_core
  PRIVATE safetensors_cpp
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

