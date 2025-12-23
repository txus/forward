add_executable(forward forward.cpp)
target_compile_features(forward PRIVATE cxx_std_26)

target_link_libraries(forward PRIVATE forward_core fmt::fmt)

add_executable(inspect inspect.cpp)
target_compile_features(inspect PRIVATE cxx_std_26)

target_link_libraries(inspect PRIVATE forward_core fmt::fmt)
