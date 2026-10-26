find_package(PkgConfig REQUIRED)
if(USE_JXL)
  pkg_check_modules(JXL REQUIRED IMPORTED_TARGET libjxl libjxl_threads)
endif()
if(USE_EXR)
  pkg_check_modules(EXR REQUIRED IMPORTED_TARGET OpenEXR)
endif()

set(ResolutionCoreSrc
  core/common.cpp
  core/geodesics.cpp
  core/geodesic_flipper.cpp
  core/layout.cpp
  core/quad_quality.cpp
  core/triangulation.cpp
  core/viewer.cpp
)

set(HierarchySrc
  hierarchy/halfedge.cpp
  hierarchy/overlay.cpp
  hierarchy/voronoi.cpp
  hierarchy/simplifier.cpp
)

set(OptimizationSrc
  optimization/field.cpp
  optimization/compression.cpp
  optimization/improvement.cpp
)

set(ExperimentalSrc
  experimental/dynamic_markings.cpp
  experimental/holonomy_coordinates.cpp
  experimental/regularization.cpp
  experimental/representatives.cpp
  experimental/simplification.cpp
  experimental/simplification_overlay.cpp
)

add_library(PennerResolutionLib
  pybind.cpp
  interface.cpp
  ${ResolutionCoreSrc}
  ${HierarchySrc}
  ${OptimizationSrc}
  ${ExperimentalSrc}
)
target_include_directories(PennerResolutionLib PUBLIC ../../include/resolution)
target_link_libraries(PennerResolutionLib PUBLIC
  PennerOptimizationLib
  PennerHolonomyLib
  PennerFeatureLib
  ${MPFR_LIBRARIES}
  igl::core
  igl::png
  Eigen3::Eigen
  geometry-central
  ${SUITESPARSE_LIBS}
  ${VISUALIZATION_LIBS}
)
if (USE_JXL)
target_link_libraries(PennerResolutionLib PUBLIC
  PkgConfig::JXL
)
endif()
if (USE_EXR)
target_link_libraries(PennerResolutionLib PUBLIC
  PkgConfig::EXR
)
endif()
if (USE_UV_OPTIMIZATION)
target_link_libraries(PennerResolutionLib PUBLIC
  SymmetricDirichletLib
)
else()
target_link_libraries(PennerResolutionLib PRIVATE
  nlohmann_json::nlohmann_json
#  OpenMP::OpenMP_CXX
)
endif()
target_compile_definitions(PennerResolutionLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
target_include_directories(PennerResolutionLib PUBLIC
  ../ext/autodiff
)

# TODO Fix for multiprecision
if ((NOT USE_MULTIPRECISION) AND (USE_WARNINGS))
    target_compile_options(PennerResolutionLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()
