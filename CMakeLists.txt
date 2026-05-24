cmake_minimum_required(VERSION 3.19..4.0)

project(
  Forward
  VERSION 0.1
  DESCRIPTION "A simple transformer inference engine"
  LANGUAGES CXX
)

# --- Compiler contract (see cmake/toolchains/) -------------------------------
# We require LLVM Clang >= 20 for C++26 + libc++. This turns a cryptic
# "cxx_std_26 not known to compiler" failure deep in configure into a clear,
# early error. CMAKE_CXX_COMPILER_ID is "Clang" for upstream/Homebrew and
# "AppleClang" for Apple's system compiler, so this also rejects AppleClang.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR
    "Forward requires LLVM Clang (got '${CMAKE_CXX_COMPILER_ID}'). "
    "Configure via a preset so the right toolchain file is used: "
    "cmake --preset macos   (or linux).")
endif()
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 20)
  message(FATAL_ERROR
    "Forward needs Clang >= 20 for C++26 (got ${CMAKE_CXX_COMPILER_VERSION}).")
endif()
# -----------------------------------------------------------------------------

# Add cmake module path for our custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Backend detection - can be overridden via -DBACKEND_CUDA=ON/OFF
if(APPLE)
  option(BACKEND_METAL "Enable Metal backend" ON)
  option(BACKEND_CUDA "Enable CUDA backend" OFF)
else()
  option(BACKEND_CUDA "Enable CUDA backend" ON)
  option(BACKEND_METAL "Enable Metal backend" OFF)
endif()

# CUDA-only feature toggles. Defaults track BACKEND_CUDA so CPU builds
# don't emit "requested but ignored" warnings.
option(FUSED_ROPE       "Use fused RoPE kernel (CUDA only)"            ${BACKEND_CUDA})
option(FUSED_ATTN       "Use fused attention kernel (CUDA only)"       ${BACKEND_CUDA})
option(SKIP_CUDA_TESTS  "Skip CUDA tests at runtime (no GPU available)" OFF)

# Propagate backend flags as compile definitions
if(BACKEND_CUDA)
  add_compile_definitions(BACKEND_CUDA)
  message(STATUS "Backend: CUDA enabled")
else()
  message(STATUS "Backend: CUDA disabled")
endif()
if(BACKEND_METAL)
  add_compile_definitions(BACKEND_METAL)

  find_library(METAL_FW        Metal        REQUIRED)
  find_library(FOUNDATION_FW   Foundation   REQUIRED)
  find_library(MPS_FW         MetalPerformanceShaders REQUIRED)

  add_library(metal_cpp INTERFACE)
  target_link_libraries(metal_cpp
  INTERFACE
    ${METAL_FW}
    ${FOUNDATION_FW}
    ${MPS_FW}
  )
  target_include_directories(metal_cpp INTERFACE
    ${CMAKE_SOURCE_DIR}/vendor/metal-cpp)

  message(STATUS "Backend: Metal enabled")
endif()

set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake/fetchcontent")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# msgpack in tokenizers_cpp is doing weird stuff
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  # Generate the per-platform .clangd so the editor parses with the same flags
  # as this build. .clangd is gitignored (per-machine). macOS uses the build/
  # compile DB directly; Linux uses the CUDA-as-clang build/clangd DB.
  if(APPLE)
    configure_file(${CMAKE_SOURCE_DIR}/.clangd.metal ${CMAKE_SOURCE_DIR}/.clangd COPYONLY)
  else()
    configure_file(${CMAKE_SOURCE_DIR}/.clangd.cuda  ${CMAKE_SOURCE_DIR}/.clangd COPYONLY)
  endif()
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

# On NixOS with Clang, clangd uses a different resource-dir than clang++
# Set the correct resource-dir for all C++ compilation
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND DEFINED CLANG_RESOURCE_DIR AND NOT CLANG_RESOURCE_DIR STREQUAL "")
  add_compile_options(-resource-dir=${CLANG_RESOURCE_DIR})
endif()


# Make dependencies available.
# tokenizers_cpp and its vendored deps (msgpack, sentencepiece) emit CMake
# deprecation + developer warnings (old cmake_minimum_required, CMP0175) that we
# can't fix upstream. Silence them only across the dependency configure, then
# restore defaults so our own src/ subdirectories still get full warnings.
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fmtlib json safetensors_cpp tokenizers_cpp)

unset(CMAKE_WARN_DEPRECATED CACHE)
unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS CACHE)

# compiled library code
add_subdirectory(src/tensor)
add_subdirectory(src/nn)
add_subdirectory(src/llama)
add_subdirectory(src/forward)

add_subdirectory(benchmarks)

# executable code
add_subdirectory(apps)

# testing only if this is the main app
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
  AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
