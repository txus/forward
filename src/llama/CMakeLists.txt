file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/llama/*.hpp")

add_library(llama STATIC model.cpp layer.cpp mlp.cpp rope.cpp)

target_link_libraries(llama
  PUBLIC tensor_core tensor_cpu nn_core nn_cpu
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

