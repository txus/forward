add_executable(forward forward.cpp)
target_compile_features(forward PRIVATE cxx_std_26)

target_link_libraries(forward PRIVATE forward_core fmt::fmt)

add_executable(inspect inspect.cpp)
target_compile_features(inspect PRIVATE cxx_std_26)

target_link_libraries(inspect PRIVATE forward_core fmt::fmt)

if(BACKEND_METAL)
  include(MetalShaders)
  add_executable(test_metal test_metal.cpp)
  target_compile_features(inspect PRIVATE cxx_std_26)
  target_link_libraries(test_metal
  PRIVATE
    fmt::fmt
    metal_cpp
  )

  compile_metal_shaders(test_metal fill.metal)

  target_include_directories(test_metal
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

endif()
