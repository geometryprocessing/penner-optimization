cmake_minimum_required(VERSION 3.25)
project(CurvatureMetric)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
set(DIR_EXT "${CMAKE_CURRENT_SOURCE_DIR}/ext")
option(USE_PYBIND "Generate pybindings" ON)
option(USE_MULTIPRECISION "Use high precision floating point" OFF)
option(ENABLE_VISUALIZATION "Generate viewers for visualization" ON)
option(CHECK_VALIDITY "Check validity pre and post conditions" ON)

find_package(Boost REQUIRED COMPONENTS system filesystem)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Options for the project
# option(USE_PYBIND "Generate pybindings" ON)
# option(USE_MULTIPRECISION "Use high precision floating point" OFF)
option(BUILD_CURVATURE_METRIC_TESTS "Build tests" ON)

# Set libigl options
option(LIBIGL_PREDICATES "Use Predicates" ON)
option(RENDER_TEXTURE "Render results" ON)
option(LIBIGL_OPENGL "Use OpenGL" ON)
option(LIBIGL_GLFW "Use GLFW" ON)
option(LIBIGL_PNG "Use PNG" ON)
option(LIBIGL_EMBREE "Use EMBREE" ON)

if(USE_MULTIPRECISION)
  # Turning this on activates mpfr in the conformal code
  option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
  add_compile_definitions(WITH_MPFR)
  include_directories(${MPFR_INCLUDE_DIR})
  include_directories(${GMP_INCLUDE_DIR})
  link_directories(${MPFR_LIBRARIES_DIR})
endif()

if(CHECK_VALIDITY)
  add_compile_definitions(CHECK_VALIDITY)
endif()

# # Add highfive library
find_package(HDF5 REQUIRED)
include(highfive)
add_library(HighFiveLib INTERFACE)
target_include_directories(HighFiveLib SYSTEM INTERFACE ${highfive_SOURCE_DIR}/include/ ${HDF5_INCLUDE_DIRS})
target_link_libraries(HighFiveLib INTERFACE ${HDF5_LIBRARIES})

# Set compile definitions
if(USE_PYBIND)
  add_compile_definitions(PYBIND)
endif()

if(USE_MULTIPRECISION)
  add_compile_definitions(MULTIPRECISION)
endif()

add_compile_definitions(USE_EMBREE)
add_compile_definitions(_USE_MATH_DEFINES)

# Get external libraries
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/util")
include(conformal_ideal_delaunay)
include(libigl)
include(spectra)

# Optionally create visualization library
if(RENDER_TEXTURE)
  add_library(visualization
    src/util/visualization.cc
  )
  target_link_libraries(visualization PUBLIC
    igl::core
    igl::glfw
    igl::png
    plot
  )
endif()

# Make main cpp library
add_subdirectory(src/optimization)

if(ENABLE_VISUALIZATION)
  add_compile_definitions(ENABLE_VISUALIZATION)
  include(polyscope)
  target_link_libraries(MetricOptimizationLib PUBLIC
    plot
    polyscope
  )
endif()

# Build pybind optimization functions
message("FIXME Building pybind libraries")

if(USE_PYBIND)
  message("Building pybind libraries")
  include(pybind11)
  add_library(optimization_py MODULE
    src/optimization/optimization_pybind.cpp
  )

  # Link visualization methods if created
  if(RENDER_TEXTURE)
    add_compile_definitions(RENDER_TEXTURE)
    target_link_libraries(optimization_py PRIVATE
      visualization
    )
  endif()

  # Link libraries
  target_link_libraries(optimization_py PUBLIC
    MetricOptimizationLib
    HighFiveLib
  )

  # Set pybinding settings
  set_target_properties(optimization_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(optimization_py PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()

if(APPLE OR NOT UNIX)
  add_subdirectory(src/app)
endif()

# Following libraries only valid for double
if(NOT USE_MULTIPRECISION)
  # Optionally build tests
  if(BUILD_CURVATURE_METRIC_TESTS)
    include(Catch2)

    # Build testing libraries
    add_library(CurvatureMetricTestsLib
      src/tests/test_area.cpp
      src/tests/test_energies.cpp
      src/tests/test_optimize.cpp
    )
    target_link_libraries(CurvatureMetricTestsLib PUBLIC
      MetricOptimizationLib
      Catch2::Catch2WithMain
    )
    set(TEST_DATA_ROOT "${PROJECT_SOURCE_DIR}/src/tests/regression/")
    target_compile_definitions(CurvatureMetricTestsLib PUBLIC TEST_DATA_DIR=\"${TEST_DATA_ROOT}\")

    # Build testing executable
    add_executable(CurvatureMetricTests
      src/tests/tests.cpp
    )
    target_link_libraries(CurvatureMetricTests PRIVATE
      CurvatureMetricTestsLib
      Catch2::Catch2WithMain
    )
  endif()
endif()
