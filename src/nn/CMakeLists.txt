file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/nn/*.hpp")

add_library(nn_core INTERFACE)

target_include_directories(nn_core
  INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(nn_core
    INTERFACE
        cxx_std_26
)

target_sources(nn_core
    INTERFACE
        ${HEADER_LIST}
)

add_subdirectory(cpu)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)

