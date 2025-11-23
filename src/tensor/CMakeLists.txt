file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/tensor/*.hpp")

FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.1.0)
FetchContent_MakeAvailable(fmtlib)
# adds fmt::fmt

add_library(tensor_core INTERFACE)

target_link_libraries(tensor_core
    INTERFACE
        fmt::fmt
)

target_include_directories(tensor_core
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(tensor_core
    INTERFACE
        cxx_std_26
)

target_sources(tensor_core
    INTERFACE
        ${HEADER_LIST}
)

add_subdirectory(cpu)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

