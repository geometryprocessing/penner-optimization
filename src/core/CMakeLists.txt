add_library(PennerOptimizationCoreLib
  area.cpp
  common.cpp
  cone_metric.cpp
  constraint.cpp
  embedding.cpp
  flip_matrix_generator.cpp
  io.cpp
  linear_algebra.cpp
  projection.cpp
  reparametrization.cpp
  shear.cpp
  vector.cpp
  vf_mesh.cpp
)
target_include_directories(PennerOptimizationCoreLib PUBLIC .)
target_link_libraries(PennerOptimizationCoreLib PUBLIC
  Eigen3::Eigen
  conformal_cpp
  igl::core
  igl::predicates
  spdlog::spdlog
  ${MPFR_LIBRARIES}
  ${POLYSCOPE_LIBRARIES}
)
target_compile_definitions(PennerOptimizationCoreLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(PennerOptimizationCoreLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)
