file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/forward/*.hpp")

FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp
  GIT_TAG 55d53aa)
FetchContent_MakeAvailable(tokenizers_cpp)

# make an automatic library - will be static or dynamic based on user setting
add_library(forward_core tokenizer.cpp loader.cpp ${HEADER_LIST})

# we need this directory, and users of our library will need it too
target_include_directories(forward_core PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(forward_core
  PUBLIC tensor_core
  PUBLIC tensor_cpu
  PUBLIC llama
  PRIVATE tokenizers_cpp
  PRIVATE safetensors_cpp
)

# all users of this library will need at least C++26
target_compile_features(forward_core PUBLIC cxx_std_26)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
