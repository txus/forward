file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/forward/*.hpp")

# make an automatic library - will be static or dynamic based on user setting
add_library(forward_core STATIC tokenizer.cpp sampler.cpp ${HEADER_LIST})

# we need this directory, and users of our library will need it too
target_include_directories(forward_core PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(forward_core
  PUBLIC llama
  PRIVATE tokenizers_cpp
)

# all users of this library will need at least C++26
target_compile_features(forward_core PUBLIC cxx_std_26)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
