set(UtilSrc
  util/boundary.cpp
  util/common.cpp
  util/embedding.cpp
  util/io.cpp
  util/linear_algebra.cpp
  util/map.cpp
  util/spanning_tree.cpp
  util/union_find.cpp
  util/vector.cpp
  util/vf_mesh.cpp
)

set(MetricSrc
  metric/area.cpp
  metric/cone_metric.cpp
  metric/constraint.cpp
  metric/flip_matrix_generator.cpp
  metric/projection.cpp
  metric/reparametrization.cpp
  metric/shear.cpp
  metric/viewer.cpp
)

set(ParametrizationSrc
  parametrization/interpolation.cpp
  parametrization/layout.cpp
  parametrization/parametrize.cpp
  parametrization/refinement.cpp
  parametrization/translation.cpp
  parametrization/triangulation.cpp
)

add_library(PennerCoreLib
  ${UtilSrc}
  ${MetricSrc}
  ${ParametrizationSrc}
)
target_include_directories(PennerCoreLib PUBLIC ../include/util)
target_include_directories(PennerCoreLib PUBLIC ../include/metric)
target_include_directories(PennerCoreLib PUBLIC ../include/parametrization)
target_link_libraries(PennerCoreLib PUBLIC 
  Eigen3::Eigen
  igl::core
  igl::predicates
  spdlog::spdlog
  conformal_cpp
  HDF5::HDF5
  #CoMISo
  ${MPFR_LIBRARIES}
  ${RENDER_LIBRARIES}
  ${SUITESPARSE_LIBS}
  ${POLYSCOPE_LIBRARIES}
)
target_compile_features(PennerCoreLib PUBLIC cxx_std_17)

target_compile_definitions(PennerCoreLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
# TODO Fix for multiprecision
if ((NOT USE_MULTIPRECISION) AND (DEBUG_WARNINGS))
    target_compile_options(PennerCoreLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()

add_subdirectory(field)
add_subdirectory(optimization)
add_subdirectory(holonomy)
add_subdirectory(feature)

add_library(PennerLib INTERFACE)
target_link_libraries(PennerLib INTERFACE
  PennerFieldLib
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
  )

  # Set pybinding settings
  set_target_properties(penner PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(penner PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()
