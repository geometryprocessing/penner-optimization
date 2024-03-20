add_executable(conformal_metric
  conformal_metric.cpp
)
target_link_libraries(conformal_metric PRIVATE
  PennerOptimizationLib
)
if(USE_SUITESPARSE)
target_link_libraries(conformal_metric PRIVATE
  SuiteSparse::SuiteSparseConfig
  SuiteSparse::CHOLMOD
)
endif()

add_executable(optimize_metric
  optimize_metric.cpp
)
target_link_libraries(optimize_metric PRIVATE
PennerOptimizationLib
)

add_executable(optimize_shear
  optimize_shear.cpp
)
target_link_libraries(optimize_shear PRIVATE
PennerOptimizationLib
)

#add_executable(plot_shear_energy
#  plot_shear_energy.cpp
#)
#target_link_libraries(plot_shear_energy PRIVATE
#  PennerOptimizationLib
#)
