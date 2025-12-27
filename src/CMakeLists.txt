add_subdirectory(util)
add_subdirectory(optimization)
add_subdirectory(holonomy)
add_subdirectory(feature)

add_library(PennerLib INTERFACE)
target_link_libraries(PennerLib INTERFACE
  PennerOptimizationLib
  PennerHolonomyLib
  PennerFeatureLib
)

if(FALSE)
#if(USE_PYBIND)
  add_library(penner MODULE
    pybind.cpp
  )

  # Link libraries
  target_link_libraries(penner PUBLIC
    PennerLib
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
