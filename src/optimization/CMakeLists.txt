add_library(MetricOptimizationLib
  area.cpp
  constraint.cpp
  convergence.cpp
  embedding.cpp
  energies.cpp
  explicit_optimization.cpp
  implicit_optimization.cpp
  interpolation.cpp
  logging.cpp
  nonlinear_optimization.cpp
  optimization_interface.cpp
  optimization_layout.cpp
  projection.cpp
  reparametrization.cpp
  shear.cpp
  targets.cpp
  transitions.cpp
  translation.cpp
)
target_include_directories(MetricOptimizationLib PUBLIC .)
target_link_libraries(MetricOptimizationLib PUBLIC
  conformal_cpp
  igl::core
  igl::predicates
  igl::glfw
  plot
  polyscope
  spdlog::spdlog
  igl::embree
  pybind11::module
  Spectra
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${MPFR_LIBRARIES}
)
target_compile_definitions(MetricOptimizationLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(MetricOptimizationLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)