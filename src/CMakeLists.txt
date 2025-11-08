# headers are optional, and do not affect add_library, but they will not show up
# in IDEs unless they are listed in add_library
file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Forward_SOURCE_DIR}/include/forward/*.hpp")

FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp
  GIT_TAG 55d53aa)
FetchContent_MakeAvailable(tokenizers_cpp)

# make an automatic library - will be static or dynamic based on user setting
add_library(forward lib.cpp tokenizer.cpp ${HEADER_LIST})

# we need this directory, and users of our library will need it too
target_include_directories(forward PUBLIC ../include)

# this depends on tokenizers_cpp
target_link_libraries(forward PRIVATE tokenizers_cpp)

# all users of this library will need at least C++26
target_compile_features(forward PUBLIC cxx_std_26)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
