set(UtilSrc
  boundary.cpp
  embedding.cpp
  io.cpp
  linear_algebra.cpp
  map.cpp
  spanning_tree.cpp
  vector.cpp
  vf_mesh.cpp
)

add_library(PennerUtilLib
  ${UtilSrc}
)
target_include_directories(PennerUtilLib PUBLIC ../../include/util)
target_link_libraries(PennerUtilLib PUBLIC
  Eigen3::Eigen
  igl::core
  igl::predicates
  spdlog::spdlog
  conformal_cpp
  ${MPFR_LIBRARIES}
  ${SUITESPARSE_LIBS}
  ${POLYSCOPE_LIBRARIES}
)
target_compile_definitions(PennerUtilLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
# TODO Fix for multiprecision
if (NOT USE_MULTIPRECISION)
    target_compile_options(PennerUtilLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()
