# headers are optional, and do not affect add_library, but they will not show up
# in IDEs unless they are listed in add_library
set(HEADER_LIST "${Forward_SOURCE_DIR}/include/forward/lib.hpp")

# make an automatic library - will be static or dynamic based on user setting
add_library(forward lib.cpp ${HEADER_LIST})

# we need this directory, and users of our library will need it too
target_include_directories(forward PUBLIC ../include)

# this depends on (header only) boost
#target_link_libraries(forward PRIVATE Boost::boost)

# all users of this library will need at least C++26
target_compile_features(forward PUBLIC cxx_std_26)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
