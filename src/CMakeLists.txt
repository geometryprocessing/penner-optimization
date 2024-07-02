add_subdirectory(optimization)
add_subdirectory(holonomy)

add_library(PennerLib INTERFACE)
target_link_libraries(PennerLib INTERFACE
  PennerOptimizationLib
  PennerHolonomyLib
)