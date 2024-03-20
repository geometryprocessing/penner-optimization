add_library(ParameterizationLib
  interpolation.cpp
  layout.cpp
  refinement.cpp
  translation.cpp
  triangulation.cpp
)
target_include_directories(ParameterizationLib PUBLIC .)
target_link_libraries(ParameterizationLib PUBLIC
  PennerOptimizationCoreLib
)
target_compile_definitions(ParameterizationLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_compile_options(ParameterizationLib PRIVATE
  -Wall -Wpedantic -Wextra -Werror
)
