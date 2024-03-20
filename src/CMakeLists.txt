add_subdirectory(core)
add_subdirectory(metric_optimization)
add_subdirectory(parameterization)
add_subdirectory(util)

add_library(PennerOptimizationLib
  penner_optimization_interface.cpp
)
target_include_directories(PennerOptimizationLib PUBLIC .)
target_link_libraries(PennerOptimizationLib PUBLIC
  PennerOptimizationCoreLib
  MetricOptimizationLib
  ParameterizationLib
  PennerOptimizationUtilLib
)
target_compile_definitions(PennerOptimizationLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(PennerOptimizationLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)
