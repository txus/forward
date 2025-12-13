cmake_minimum_required(VERSION 3.19..4.0)

project(
  Forward
  VERSION 0.1
  DESCRIPTION "A simple transformer inference engine"
  LANGUAGES CXX
)

set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake/fetchcontent")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# msgpack in tokenizers_cpp is doing weird stuff
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_CXX_EXTENSIONS ON)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  include(CTest)

  find_package(Doxygen)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()
endif()

include(FetchContent)

# Declare all external dependencies in one place
FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.1.0
)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.12.0
)

FetchContent_Declare(
  safetensors_cpp
  GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp.git
  GIT_TAG 10f7d8f
)
set(SAFETENSORS_CPP_CXX_EXCEPTIONS ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp
  GIT_TAG 55d53aa
)

# Make dependencies available
FetchContent_MakeAvailable(fmtlib json safetensors_cpp tokenizers_cpp)

# compiled library code
add_subdirectory(src/tensor)
add_subdirectory(src/nn)
add_subdirectory(src/llama)
add_subdirectory(src/forward)

# executable code
add_subdirectory(apps)

# testing only if this is the main app
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
  AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

