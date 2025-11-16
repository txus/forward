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

# compiled library code
add_subdirectory(src/tensor)
add_subdirectory(src/llama)
add_subdirectory(src/forward)

# executable code
add_subdirectory(apps)

# testing only if this is the main app
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
  AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

