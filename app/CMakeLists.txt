add_library(ApplicationUtilLib INTERFACE)
target_include_directories(ApplicationUtilLib INTERFACE .)
target_link_libraries(ApplicationUtilLib INTERFACE
  CLI11::CLI11
)

add_executable(generate_field
generate_field.cpp
)
target_link_libraries(generate_field PRIVATE
  PennerLib
  ApplicationUtilLib
)

add_executable(generate_aligned_field
generate_aligned_field.cpp
)
target_link_libraries(generate_aligned_field PRIVATE
  PennerLib
  ApplicationUtilLib
)

add_executable(parametrize_seamless
  parametrize_seamless.cpp
)
target_link_libraries(parametrize_seamless PRIVATE
  PennerLib
  SymmetricDirichletLib
  CLI11::CLI11
)

add_executable(parameterize_aligned
parameterize_aligned.cpp
)
target_link_libraries(parameterize_aligned PRIVATE
  PennerLib
  SymmetricDirichletLib
  ApplicationUtilLib
)

add_executable(test_interface
test_interface.cpp
)
target_link_libraries(test_interface PRIVATE
  PennerLib
  SymmetricDirichletLib
  ApplicationUtilLib
)

add_executable(optimize_metric
  optimize_metric.cpp
)
target_link_libraries(optimize_metric PRIVATE
  PennerLib
  CLI11::CLI11
)

# add_executable(optimize_shear
#   optimize_shear.cpp
# )
# target_link_libraries(optimize_shear PRIVATE
#   PennerLib
#   CLI11::CLI11
# )

# only build visualization executables if visualization enabled
if (NOT USE_MULTIPRECISION)
if (ENABLE_VISUALIZATION)
    add_executable(view_quad_mesh
    view_quad_mesh.cpp
    )
    target_link_libraries(view_quad_mesh PRIVATE
      PennerLib
      ApplicationUtilLib
    )

    add_executable(view_seamless
    view_seamless.cpp
    )
    target_link_libraries(view_seamless PRIVATE
      PennerLib
      ApplicationUtilLib
    )
endif()
endif()
