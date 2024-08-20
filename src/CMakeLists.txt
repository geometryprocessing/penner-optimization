add_subdirectory(util)
add_subdirectory(optimization)
add_subdirectory(holonomy)

add_library(PennerLib INTERFACE)
target_link_libraries(PennerLib INTERFACE
  PennerOptimizationLib
  PennerHolonomyLib
)

if(USE_PYBIND)
  add_library(penner MODULE
    pybind.cpp
  )

  # Link libraries
  target_link_libraries(penner PUBLIC
    PennerOptimizationLib
    PennerHolonomyLib
    ${RENDER_LIBRARIES}
  )

  # Set pybinding settings
  set_target_properties(penner PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(penner PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()