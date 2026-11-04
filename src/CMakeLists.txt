set(UtilSrc
  util/boundary.cpp
  util/common.cpp
  util/embedding.cpp
  util/io.cpp
  util/linear_algebra.cpp
  util/map.cpp
  util/quads.cpp
  util/spanning_tree.cpp
  util/union_find.cpp
  util/vector.cpp
  util/vf_mesh.cpp
)

set(MetricSrc
  metric/area.cpp
  metric/barycentric.cpp
  metric/cone_metric.cpp
  metric/constraint.cpp
  metric/flip_matrix_generator.cpp
  metric/interface.cpp
  metric/projection.cpp
  metric/quality.cpp
  metric/shear.cpp
  metric/viewer.cpp
)

set(ParametrizationSrc
  parametrization/interpolation.cpp
  parametrization/error.cpp
  parametrization/layout.cpp
  parametrization/parametrize.cpp
  parametrization/refinement.cpp
  parametrization/translation.cpp
  parametrization/triangulation.cpp
)

add_library(PennerCoreLib
  ${UtilSrc}
  ${MetricSrc}
  ${ParametrizationSrc}
)
target_include_directories(PennerCoreLib PUBLIC ../include/util)
target_include_directories(PennerCoreLib PUBLIC ../include/metric)
target_include_directories(PennerCoreLib PUBLIC ../include/parametrization)
target_link_libraries(PennerCoreLib PUBLIC 
  Eigen3::Eigen
  igl::core
  igl::predicates
  spdlog::spdlog
  conformal_cpp
  HDF5::HDF5
  #CoMISo
  ${MPFR_LIBRARIES}
  ${RENDER_LIBRARIES}
  ${SUITESPARSE_LIBS}
  ${POLYSCOPE_LIBRARIES}
)
target_compile_features(PennerCoreLib PUBLIC cxx_std_17)

target_compile_definitions(PennerCoreLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
# TODO Fix for multiprecision
if ((NOT USE_MULTIPRECISION) AND (DEBUG_WARNINGS))
    target_compile_options(PennerCoreLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()

# Must stay in dependency order
set(PENNER_ALL_LIBS Field Optimization Holonomy Feature Resolution)
set(PENNER_Holonomy_DEPS   Field Optimization)
set(PENNER_Feature_DEPS    Field Optimization Holonomy)
set(PENNER_Resolution_DEPS Field Optimization Holonomy Feature)

foreach(lib IN LISTS PENNER_LIBS)
  if(NOT lib IN_LIST PENNER_ALL_LIBS)
    message(FATAL_ERROR "Unknown entry '${lib}' in PENNER_LIBS. Valid: ${PENNER_ALL_LIBS}")
  endif()
endforeach()

# Add dependencies: walking in reverse order means one pass is enough
set(PENNER_ENABLED_LIBS ${PENNER_LIBS})
set(_reverse ${PENNER_ALL_LIBS})
list(REVERSE _reverse)
foreach(lib IN LISTS _reverse)
  if(lib IN_LIST PENNER_ENABLED_LIBS)
    list(APPEND PENNER_ENABLED_LIBS ${PENNER_${lib}_DEPS})
  endif()
endforeach()
list(REMOVE_DUPLICATES PENNER_ENABLED_LIBS)
message(STATUS "Penner libraries: requested [${PENNER_LIBS}], building [${PENNER_ENABLED_LIBS}]")

add_library(PennerLib INTERFACE)
target_link_libraries(PennerLib INTERFACE PennerCoreLib)

foreach(lib IN LISTS PENNER_ALL_LIBS)
  if(lib IN_LIST PENNER_ENABLED_LIBS)
    string(TOLOWER ${lib} dir)
    string(TOUPPER ${lib} upper)
    add_subdirectory(${dir})
    target_compile_definitions(PennerLib INTERFACE PENNER_WITH_${upper})
    if(lib IN_LIST PENNER_LIBS)
      target_link_libraries(PennerLib INTERFACE Penner${lib}Lib)
    endif()
  endif()
endforeach()

if(USE_PYBIND)
  add_library(penner MODULE
    pybind.cpp
  )

  # Link libraries
  target_link_libraries(penner PUBLIC
    PennerLib
  )

  # set sources for pybind
  foreach(lib IN LISTS PENNER_ENABLED_LIBS)
    string(TOLOWER ${lib} dir)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/pybind.cpp)
      target_sources(penner PRIVATE ${dir}/pybind.cpp)
    endif()
  endforeach()

  # Set pybinding settings
  set_target_properties(penner PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/py
  )
  set_target_properties(penner PROPERTIES PREFIX
    "${PYTHON_MODULE_PREFIX}"
  )
endif()
