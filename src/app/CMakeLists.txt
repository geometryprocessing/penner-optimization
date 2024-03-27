add_executable(optimize_metric
  optimize_metric.cpp
)
target_link_libraries(optimize_metric PRIVATE
  PennerOptimizationLib
  CLI11::CLI11
)

add_executable(optimize_shear
  optimize_shear.cpp
)
target_link_libraries(optimize_shear PRIVATE
  PennerOptimizationLib
  CLI11::CLI11
)

add_executable(plot_shear_energy
  plot_shear_energy.cpp
)
target_link_libraries(plot_shear_energy PRIVATE
  PennerOptimizationLib
)
