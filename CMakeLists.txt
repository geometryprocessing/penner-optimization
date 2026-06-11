cmake_minimum_required(VERSION 3.25)
project(PennerOptimization)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
add_compile_options(-Wno-register)
set(DIR_EXT "${CMAKE_CURRENT_SOURCE_DIR}/ext")

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set options if top level
if (PROJECT_IS_TOP_LEVEL)
  # Options to generate python bindings
  option(USE_PYBIND "Generate pybindings" ON)

  # Alternative library options
  option(USE_MULTIPRECISION "Use high precision floating point" OFF)
  option(USE_SUITESPARSE "Use suite sparse methods for matrix inversion" ON)
  option(USE_COMISO "Use Comiso for field generation" ON)
  
  option(USE_UV_OPTIMIZATION "Add uv optimization library" ON)

  # Visualization options
  option(ENABLE_VISUALIZATION "Generate viewers for visualization" ON)
  option(RENDER_TEXTURE "Render results" ON)

  # Validity check options
  option(BUILD_CURVATURE_METRIC_TESTS "Build tests" ON)
  option(CHECK_VALIDITY "Check validity pre and post conditions" ON)


  # Set default libigl and suitesparse options
  option(LIBIGL_PREDICATES "Use Predicates" ON)
  #set ( SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;cholmod;spqr" )
  option ( SUITESPARSE_USE_CUDA OFF )
endif()

set ( SUITESPARSE_ENABLE_PROJECTS "${SUITESPARSE_ENABLE_PROJECTS};suitesparse_config;cholmod" CACHE STRING "" FORCE)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(eigen)

include(boost)
# create a fake BoostConfig.cmake before libigl runs and force Boost_DIR
# to it so ignore_package (which uses no FORCE) cannot override it.
set(_fake_boost_dir "${CMAKE_BINARY_DIR}/cmake/FakeBoost")
file(MAKE_DIRECTORY "${_fake_boost_dir}")
file(WRITE "${_fake_boost_dir}/BoostConfig.cmake"
    "set(Boost_FOUND TRUE)\nset(Boost_system_FOUND TRUE)\nset(Boost_thread_FOUND TRUE)\n")
file(WRITE "${_fake_boost_dir}/BoostConfigVersion.cmake"
    "set(PACKAGE_VERSION 1.83.0)\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\nset(PACKAGE_VERSION_EXACT FALSE)\n")
set(Boost_DIR "${_fake_boost_dir}" CACHE PATH "" FORCE)

if(BUILD_CURVATURE_METRIC_TESTS)
  include(Catch2)
  include(CTest)
endif()

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
  set(LIBIGL_COPYLEFT_CORE   ON CACHE BOOL "" FORCE)
  #set(LIBIGL_COPYLEFT_COMISO   ON CACHE BOOL "" FORCE)
  #option(LIBIGL_COPYLEFT_COMISO "Use COSIMO" ON)
  #set(COMISO_ENABLE_SUITESPARSE_DEFAULT OFF)
  set(COMISO_NO_INSTALL ON)
  set(COMISO_LIBS
    #igl_copyleft::comiso
    CoMISo
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
  list(APPEND SUITESPARSE_LIBS
    SuiteSparse::SuiteSparseConfig
    #SuiteSparse::SPQR
    SuiteSparse::CHOLMOD
  )
endif()

# Get external libraries
include(conformal_ideal_delaunay)
include(libigl)
include(spectra)
include(cli11)
include(geometry-central)
include(comiso)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# get optimization library
if (USE_UV_OPTIMIZATION)
    add_compile_definitions(USE_UV_OPTIMIZATION)
    include(symmetric-dirichlet)
else()
    include(json)
endif()

include(directional)
add_library(DirectionalLib INTERFACE)
target_include_directories(DirectionalLib SYSTEM INTERFACE 
  ${directional_SOURCE_DIR}/include
)
target_compile_features(DirectionalLib INTERFACE cxx_std_20)


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
