set(HolonomyCoreSrc
  core/boundary_basis.cpp
  core/common.cpp
  core/dual_lengths.cpp
  core/dual_loop.cpp
  core/dual_segment.cpp
  core/field.cpp
  core/forms.cpp
  core/homology_basis.cpp
  core/intrinsic_field.cpp
  core/quality.cpp
  core/viewer.cpp
)

set(HolonomySrc
  holonomy/cones.cpp
  holonomy/constraint.cpp
  holonomy/holonomy.cpp
  holonomy/marked_penner_cone_metric.cpp
  holonomy/newton.cpp
  holonomy/rotation_form.cpp
)

set(SimilaritySrc
  similarity/conformal.cpp
  similarity/constraint.cpp
  similarity/energy.cpp
  similarity/layout.cpp
  similarity/similarity_penner_cone_metric.cpp
)

add_library(PennerHolonomyLib
  holonomy_interface.cpp
  ${HolonomyCoreSrc}
  ${HolonomySrc}
  ${SimilaritySrc}
  ${DirichletSrc}
)
target_include_directories(PennerHolonomyLib PUBLIC ../../include/holonomy)
target_link_libraries(PennerHolonomyLib PUBLIC
  PennerUtilLib
  PennerOptimizationLib
  ${VISUALIZATION_LIBS}
)
target_link_libraries(PennerHolonomyLib PRIVATE
  geometry-central
  nlohmann_json::nlohmann_json
  ${COMISO_LIBS}
)
target_compile_definitions(PennerHolonomyLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
# TODO Fix for multiprecision
if (NOT USE_MULTIPRECISION)
    target_compile_options(PennerHolonomyLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()

if(USE_PYBIND)
  add_library(holonomy_py MODULE
    holonomy_pybind.cpp
  )

  # Link libraries
  target_link_libraries(holonomy_py PUBLIC
    PennerHolonomyLib
    pybind11::module
    ${RENDER_LIBRARIES}
  )

  # Set pybinding settings
  set_target_properties(holonomy_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(holonomy_py PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()

