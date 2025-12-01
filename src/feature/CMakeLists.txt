set(FeatureCoreSrc
  core/boundary_path.cpp
  core/common.cpp
  core/component_mesh.cpp
  core/io.cpp
  core/union_meshes.cpp
  core/quads.cpp
  core/viewer.cpp
)

set(DirichletSrc
  dirichlet/angle_constraint_relaxer.cpp
  dirichlet/cone_perturber.cpp
  dirichlet/constraint.cpp
  dirichlet/dirichlet_penner_cone_metric.cpp
  dirichlet/optimization.cpp
)

set(FeatureSrc
  feature/error.cpp
  feature/features.cpp
  feature/gluing.cpp
)

set(SurgerySrc
  surgery/cut_mesh_layout.cpp
  surgery/cut_metric_generator.cpp
  surgery/refinement.cpp
  surgery/stitching.cpp
)

set(FeatureUtilSrc
  util/union_find.cpp
)

set(FeatureExperimentalSrc
  experimental/boundary_constraint_generator.cpp
  experimental/boundary_constraint_validator.cpp
  #experimental/newton.cpp
  experimental/patches.cpp
  experimental/transfer.cpp
  experimental/energy.cpp
  experimental/junctions.cpp
  experimental/polygon_cones.cpp
  experimental/marked_refinement.cpp
)

add_library(PennerFeatureLib
  interface.cpp
  pybind.cpp
  ${FeatureCoreSrc}
  ${DirichletSrc}
  ${FeatureSrc}
  ${SurgerySrc}
  ${FeatureUtilSrc}
  ${FeatureExperimentalSrc}
)
target_include_directories(PennerFeatureLib PUBLIC ../../include/feature)
target_link_libraries(PennerFeatureLib PUBLIC
  PennerOptimizationLib
  PennerHolonomyLib
  ${MPFR_LIBRARIES}
  igl::core
  ${SUITESPARSE_LIBS}
  ${VISUALIZATION_LIBS}
)
target_link_libraries(PennerFeatureLib PRIVATE
  nlohmann_json::nlohmann_json
)
target_compile_definitions(PennerFeatureLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
# TODO Fix for multiprecision
if ((NOT USE_MULTIPRECISION) AND (USE_WARNINGS))
    target_compile_options(PennerFeatureLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()

if(USE_PYBIND)
  add_library(feature_py MODULE
    pybind.cpp
  )

  # Link libraries
  target_link_libraries(feature_py PUBLIC
    PennerFeatureLib
    pybind11::module
  )

  # Set pybinding settings
  set_target_properties(feature_py PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()

