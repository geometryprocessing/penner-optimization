add_library(PennerOptimizationUtilLib
  shapes.cpp
  viewers.cpp
)
target_include_directories(PennerOptimizationUtilLib PUBLIC .)
target_link_libraries(PennerOptimizationUtilLib PUBLIC
  PennerOptimizationCoreLib
)
target_compile_definitions(PennerOptimizationUtilLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(PennerOptimizationUtilLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)
