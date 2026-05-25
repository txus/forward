function(compile_metal_shaders TARGET_NAME)
  set(METAL_SOURCES ${ARGN})
  set(AIR_FILES "")

  foreach(SRC ${METAL_SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    set(AIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air)
    add_custom_command(
      OUTPUT ${AIR}
      COMMAND xcrun -sdk macosx metal
              -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}
              -o ${AIR}
              -std=metal3.1 -O3
      DEPENDS ${SRC}
      COMMENT "Compiling Metal shader ${SRC}"
    )
    list(APPEND AIR_FILES ${AIR})
  endforeach()

  set(METALLIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
  add_custom_command(
    OUTPUT ${METALLIB}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking Metal library"
  )

  add_custom_target(${TARGET_NAME}_shaders DEPENDS ${METALLIB})
  add_dependencies(${TARGET_NAME} ${TARGET_NAME}_shaders)

  target_compile_definitions(${TARGET_NAME}
    PRIVATE METAL_DEFAULT_LIBRARY_PATH="${METALLIB}")
endfunction()
