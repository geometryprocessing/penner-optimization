cmake_minimum_required(VERSION 3.25)
project(PennerOptimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
set(DIR_EXT "${CMAKE_CURRENT_SOURCE_DIR}/ext")

# Set options if top level
if (PROJECT_IS_TOP_LEVEL)
  # Options for the project
  option(USE_MULTIPRECISION "Use high precision floating point" OFF)
  option(CHECK_VALIDITY "Check validity pre and post conditions" ON)
  option(ENABLE_VISUALIZATION "Generate viewers for visualization" ON)
  option(RENDER_TEXTURE "Render results" ON)
  option(USE_SUITESPARSE "Use suite sparse methods for matrix inversion" ON)
  option(USE_PYBIND "Generate pybindings" ON)
  option(BUILD_CURVATURE_METRIC_TESTS "Build tests" ON)

  # Set libigl and suitesparse options
  option(LIBIGL_PREDICATES "Use Predicates" ON)
  set ( SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;cholmod;spqr" )
  option ( SUITESPARSE_USE_CUDA OFF )
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Optionally get multiprecision libraries
if(USE_MULTIPRECISION)
  # Turning this on activates mpfr in the conformal code
  option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
  add_compile_definitions(WITH_MPFR)
  find_package(MPFR REQUIRED QUIET)
  include_directories(${MPFR_INCLUDE_DIR})
  include_directories(${GMP_INCLUDE_DIR})
  link_directories(${MPFR_LIBRARIES_DIR})
endif()

# Set compile definitions
add_compile_definitions(USE_EMBREE)
add_compile_definitions(_USE_MATH_DEFINES)
if(USE_MULTIPRECISION)
  add_compile_definitions(MULTIPRECISION)
endif()
if(CHECK_VALIDITY)
  add_compile_definitions(CHECK_VALIDITY)
endif()
if (RENDER_TEXTURE)
  option(LIBIGL_OPENGL "Use OpenGL" ON)
  option(LIBIGL_GLFW "Use GLFW" ON)
  option(LIBIGL_PNG "Use PNG" ON)
  option(LIBIGL_EMBREE "Use EMBREE" ON)
  add_compile_definitions(RENDER_TEXTURE)
endif()
if(USE_PYBIND)
  add_compile_definitions(PYBIND)
endif()

# Set suitesparse compile definitions
# WARNING: This compile definition publicly links suitesparse into the
# conformal ideal delaunay library
if(USE_SUITESPARSE)
  add_compile_definitions(USE_SUITESPARSE)
endif()

# Get external libraries
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/util")
include(conformal_ideal_delaunay)
include(libigl)
include(spectra)
include(cli11)

# Optionally create visualization library
if(RENDER_TEXTURE)
  add_library(rendering
    src/util/visualization.cc
  )
  target_link_libraries(rendering PUBLIC
    igl::core
    igl::glfw
    igl::png
    igl::embree
    plot
  )
  set(RENDER_LIBRARIES
    rendering
  )
endif()

# Optionally enable polyscope visualization
if(ENABLE_VISUALIZATION)
  add_compile_definitions(ENABLE_VISUALIZATION)
  include(polyscope)
  set(POLYSCOPE_LIBRARIES
    polyscope
  )
endif()

# Install executables to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Make main cpp library
add_subdirectory(src)

# Build executables
add_subdirectory(src/app)

# Build pybind optimization functions
if(USE_PYBIND)
  include(pybind11)
  add_library(optimization_py MODULE
    src/penner_optimization_pybind.cpp
  )

  # Link libraries
  target_link_libraries(optimization_py PUBLIC
    PennerOptimizationLib
    pybind11::module
    ${RENDER_LIBRARIES}
  )

  # Set pybinding settings
  set_target_properties(optimization_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(optimization_py PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()

# Optionally build tests (only valid for double precision)
if((BUILD_CURVATURE_METRIC_TESTS) AND (NOT USE_MULTIPRECISION))
  include(Catch2)

  # Build testing executable
  add_executable(CurvatureMetricTests
    src/tests/tests.cpp
  )
  target_link_libraries(CurvatureMetricTests PRIVATE
    PennerOptimizationLib
    Catch2::Catch2WithMain
  )
  set(TEST_DATA_ROOT "${PROJECT_SOURCE_DIR}/src/tests/regression/")
  target_compile_definitions(CurvatureMetricTests PUBLIC TEST_DATA_DIR=\"${TEST_DATA_ROOT}\")
endif()
