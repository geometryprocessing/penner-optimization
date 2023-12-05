add_library(MetricOptimizationLib
  area.cpp
  constraint.cpp
  convergence.cpp
  delaunay.cpp
  embedding.cpp
  energies.cpp
  energy_functor.cpp
#  explicit_optimization.cpp
  implicit_optimization.cpp
  interpolation.cpp
  layout.cpp
  logging.cpp
  nonlinear_optimization.cpp
  optimization_interface.cpp
  projection.cpp
  refinement.cpp
  reparametrization.cpp
  shear.cpp
  targets.cpp
  translation.cpp
  triangulation.cpp
  viewers.cpp
)
target_include_directories(MetricOptimizationLib PUBLIC .)
target_link_libraries(MetricOptimizationLib PUBLIC
  conformal_cpp
  igl::core
  igl::predicates
  igl::glfw
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