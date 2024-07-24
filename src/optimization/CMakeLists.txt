set(OptimizationCoreSrc
  core/area.cpp
  core/common.cpp
  core/cone_metric.cpp
  core/constraint.cpp
  core/flip_matrix_generator.cpp
  core/projection.cpp
  core/reparametrization.cpp
  core/shear.cpp
)

set(MetricOptimizationSrc
  metric_optimization/convergence.cpp
  metric_optimization/energies.cpp
  metric_optimization/energy_functor.cpp
  metric_optimization/energy_weights.cpp
  metric_optimization/explicit_optimization.cpp
  metric_optimization/implicit_optimization.cpp
  metric_optimization/nonlinear_optimization.cpp
)

set(ParameterizationSrc
  parameterization/interpolation.cpp
  parameterization/layout.cpp
  parameterization/refinement.cpp
  parameterization/translation.cpp
  parameterization/triangulation.cpp
)

set(OptimizationUtilSrc
  util/shapes.cpp
  util/viewers.cpp
)

add_library(PennerOptimizationLib
  interface.cpp
  ${OptimizationCoreSrc}
  ${MetricOptimizationSrc}
  ${ParameterizationSrc}
  ${OptimizationUtilSrc}
)
target_include_directories(PennerOptimizationLib PUBLIC ../../include/optimization)
target_link_libraries(PennerOptimizationLib PUBLIC
  PennerUtilLib
)
target_compile_definitions(PennerOptimizationLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(PennerOptimizationLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)

if(USE_PYBIND)
  add_library(optimization_py MODULE
    pybind.cpp
  )

  # Link libraries
  target_link_libraries(optimization_py PUBLIC
    PennerOptimizationLib
    pybind11::module
    ${RENDER_LIBRARIES}
  )

  # Set pybinding settings
  set_target_properties(optimization_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(optimization_py PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()
