
set(MetricOptimizationSrc
  metric_optimization/energies.cpp
  metric_optimization/energy_functor.cpp
  metric_optimization/energy_weights.cpp
  metric_optimization/explicit_optimization.cpp
  metric_optimization/implicit_optimization.cpp
  metric_optimization/nonlinear_optimization.cpp
)

set(OptimizationUtilSrc
  util/shapes.cpp
)

add_library(PennerOptimizationLib
  interface.cpp
  pybind.cpp
  ${MetricOptimizationSrc}
  ${OptimizationUtilSrc}
)
target_include_directories(PennerOptimizationLib PUBLIC ../../include/optimization)
target_link_libraries(PennerOptimizationLib PUBLIC
  PennerCoreLib
  ${PYBIND_LIBRARIES}
)
target_compile_definitions(PennerOptimizationLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
if ((NOT USE_MULTIPRECISION) AND (DEBUG_WARNINGS))
    target_compile_options(PennerOptimizationLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()

if(BUILD_CURVATURE_METRIC_TESTS)
  add_executable(penner_tests tests/tests.cpp)
  target_link_libraries(penner_tests PRIVATE PennerOptimizationLib Catch2::Catch2WithMain)
  target_include_directories(penner_tests PRIVATE tests)
  target_compile_definitions(penner_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/regression"
  )
  add_test(NAME penner_tests COMMAND penner_tests)
endif()
