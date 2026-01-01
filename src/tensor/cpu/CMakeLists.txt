file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

add_library(tensor_cpu STATIC ops.cpp loader.cpp ../common/safetensors_impl.cpp)

# Find and enable OpenMP (required)
# On macOS with Homebrew, help CMake find libomp
if(APPLE)
  if(EXISTS /opt/homebrew/opt/libomp)
    set(OpenMP_ROOT /opt/homebrew/opt/libomp)
  elseif(EXISTS /usr/local/opt/libomp)
    set(OpenMP_ROOT /usr/local/opt/libomp)
  endif()
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(tensor_cpu PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(tensor_cpu
  PUBLIC tensor_core
  PRIVATE safetensors_cpp
)

target_include_directories(tensor_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

