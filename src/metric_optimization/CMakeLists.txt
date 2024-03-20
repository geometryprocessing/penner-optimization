add_library(MetricOptimizationLib
  convergence.cpp
  energies.cpp
  energy_functor.cpp
  energy_weights.cpp
  explicit_optimization.cpp
  implicit_optimization.cpp
  nonlinear_optimization.cpp
)
target_include_directories(MetricOptimizationLib PUBLIC .)
target_link_libraries(MetricOptimizationLib PUBLIC
  PennerOptimizationCoreLib
)
target_compile_definitions(MetricOptimizationLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(MetricOptimizationLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)
