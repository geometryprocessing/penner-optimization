cmake_minimum_required(VERSION 3.25)
project(PennerOptimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
set(DIR_EXT "${CMAKE_CURRENT_SOURCE_DIR}/ext")

if (PROJECT_IS_TOP_LEVEL)
  # Options for the project
  option(USE_PYBIND "Generate pybindings" ON)
  option(USE_MULTIPRECISION "Use high precision floating point" OFF)
  option(ENABLE_VISUALIZATION "Generate viewers for visualization" ON)
  option(CHECK_VALIDITY "Check validity pre and post conditions" ON)
  option(BUILD_CURVATURE_METRIC_TESTS "Build tests" ON)
  option(USE_SUITESPARSE "Use suite sparse methods for matrix inversion" OFF)

  # Set libigl options
  option(LIBIGL_PREDICATES "Use Predicates" ON)
  option(RENDER_TEXTURE "Render results" OFF)
  set ( SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;cholmod;spqr" )
endif()

#find_package(Boost REQUIRED COMPONENTS system filesystem serialization)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(USE_MULTIPRECISION)
  # Turning this on activates mpfr in the conformal code
  option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
  add_compile_definitions(WITH_MPFR)
  find_package(MPFR REQUIRED QUIET)
  include_directories(${MPFR_INCLUDE_DIR})
  include_directories(${GMP_INCLUDE_DIR})
  link_directories(${MPFR_LIBRARIES_DIR})
endif()

if(CHECK_VALIDITY)
  add_compile_definitions(CHECK_VALIDITY)
endif()

if (RENDER_TEXTURE)
  option(LIBIGL_OPENGL "Use OpenGL" ON)
  option(LIBIGL_GLFW "Use GLFW" ON)
  option(LIBIGL_PNG "Use PNG" ON)
  option(LIBIGL_EMBREE "Use EMBREE" ON)
endif()

# # Add highfive library
if (USE_HIGHFIVE)
  find_package(HDF5 REQUIRED)
  include(highfive)
  add_library(HighFiveLib INTERFACE)
  target_include_directories(HighFiveLib SYSTEM INTERFACE ${highfive_SOURCE_DIR}/include/ ${HDF5_INCLUDE_DIRS})
  target_link_libraries(HighFiveLib INTERFACE ${HDF5_LIBRARIES})
  add_compile_definitions(USE_HIGHFIVE)
endif()

# Set compile definitions
if(USE_PYBIND)
  add_compile_definitions(PYBIND)
endif()

if(USE_MULTIPRECISION)
  add_compile_definitions(MULTIPRECISION)
endif()

if(USE_SUITESPARSE)
  add_compile_definitions(USE_SUITESPARSE)
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
    igl::embree
    plot
  )
endif()

if(ENABLE_VISUALIZATION)
  add_compile_definitions(ENABLE_VISUALIZATION)
endif()

if(ENABLE_VISUALIZATION)
  include(polyscope)
  set(POLYSCOPE_LIBRARIES
    polyscope
  )
endif()

# Make main cpp library
add_subdirectory(src)


if(USE_SUITESPARSE)
  include(suitesparse)
  target_link_libraries(PennerOptimizationLib PRIVATE
    SuiteSparse::SuiteSparseConfig
    SuiteSparse::CHOLMOD
  )
endif()

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
  )
  # Link visualization methods if created
  if(RENDER_TEXTURE)
    add_compile_definitions(RENDER_TEXTURE)
    target_link_libraries(optimization_py PRIVATE
      visualization
    )
  endif()
  if(USE_HIGHFIVE)
    target_link_libraries(optimization_py PUBLIC
      HighFiveLib
    )
  endif()

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
    #add_library(CurvatureMetricTestsLib
    #)
    #target_link_libraries(CurvatureMetricTestsLib PUBLIC
    #  PennerOptimizationLib
    #  Catch2::Catch2WithMain
    #)

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
endif()
