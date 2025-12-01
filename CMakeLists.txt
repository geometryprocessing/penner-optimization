cmake_minimum_required(VERSION 3.30)
project(PennerOptimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
add_compile_options(-Wno-register)
set(DIR_EXT "${CMAKE_CURRENT_SOURCE_DIR}/ext")

# Set options if top level
if (PROJECT_IS_TOP_LEVEL)
  # Options to generate python bindings
  option(USE_PYBIND "Generate pybindings" ON)

  # Alternative library options
  option(USE_MULTIPRECISION "Use high precision floating point" OFF)
  option(USE_SUITESPARSE "Use suite sparse methods for matrix inversion" ON)
  option(USE_COMISO "Use Comiso for field generation" ON)

  # Visualization options
  option(ENABLE_VISUALIZATION "Generate viewers for visualization" ON)
  option(RENDER_TEXTURE "Render results" ON)

  # Validity check options
  option(BUILD_CURVATURE_METRIC_TESTS "Build tests" ON)
  option(CHECK_VALIDITY "Check validity pre and post conditions" ON)

  # Set default libigl and suitesparse options
  option(LIBIGL_PREDICATES "Use Predicates" ON)
  set ( SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;cholmod;spqr" )
  option ( SUITESPARSE_USE_CUDA OFF )
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(boost)
#include(eigen)

# Optionally get multiprecision libraries
if(USE_MULTIPRECISION OR USE_MULTIPRECISION_OVERLAY)
  # Turning this on activates mpfr in the conformal code
  option(LIBIGL_COPYLEFT_CGAL "Use CGAL" ON)
  add_compile_definitions(WITH_MPFR)
  find_package(MPFR REQUIRED QUIET)
  include_directories(${MPFR_INCLUDE_DIR})
  include_directories(${GMP_INCLUDE_DIR})
  link_directories(${MPFR_LIBRARIES_DIR})
endif()
if(USE_MULTIPRECISION) 
  add_compile_definitions(MULTIPRECISION)
endif()

# Optionally get frame field libraries
if (USE_COMISO)
  add_compile_definitions(USE_COMISO)
  option(LIBIGL_COPYLEFT_COMISO "Use COSIMO" ON)
  set(COMISO_LIBS
    igl_copyleft::comiso
  )
endif()

# Set libigl options for rendering if enabled
if (RENDER_TEXTURE)
  option(LIBIGL_OPENGL "Use OpenGL" ON)
  option(LIBIGL_GLFW "Use GLFW" ON)
  option(LIBIGL_PNG "Use PNG" ON)
endif()

# Set compile definitions
add_compile_definitions(_USE_MATH_DEFINES)
if(USE_MULTIPRECISION)
  add_compile_definitions(MULTIPRECISION)
endif()
if(CHECK_VALIDITY)
  add_compile_definitions(CHECK_VALIDITY)
endif()
if (RENDER_TEXTURE)
  add_compile_definitions(RENDER_TEXTURE)
endif()
if(USE_PYBIND)
  add_compile_definitions(PYBIND)
endif()
# WARNING: This compile definition publicly links suitesparse into the
# conformal ideal delaunay library
if(USE_SUITESPARSE)
  add_compile_definitions(USE_SUITESPARSE)
  set(SUITESPARSE_LIBS
    SuiteSparse::SuiteSparseConfig
    SuiteSparse::SPQR
    SuiteSparse::CHOLMOD
  )
endif()

# Get external libraries
include(boost)
include(conformal_ideal_delaunay)
include(libigl)
include(spectra)
include(cli11)
include(json)
include(geometry-central)
#include(symmetric-dirichlet)

include(directional)
add_library(DirectionalLib INTERFACE)
target_include_directories(DirectionalLib SYSTEM INTERFACE 
  ${directional_SOURCE_DIR}/include
)

# Optionally create rendering library
if(RENDER_TEXTURE)
  # Build core rendering library
  # TODO: Move somewhere reasonable
  add_library(rendering
    src/optimization/util/visualization.cc
  )
  target_include_directories(rendering PUBLIC include/optimization/optimization/util)
  target_link_libraries(rendering PUBLIC
    igl::core
    igl::glfw
    igl::png
    plot
  )


  # Change rendering libraries from null to singleton
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

# Optionally build pybind 
if(USE_PYBIND)
  include(pybind11)
  set(PYBIND_LIBRARIES
    pybind11::module
  )
endif()

# Install executables to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Make main cpp library
add_subdirectory(src)
