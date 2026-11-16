add_library(ApplicationUtilLib INTERFACE)
target_include_directories(ApplicationUtilLib INTERFACE .)
target_link_libraries(ApplicationUtilLib INTERFACE
)

function(penner_add_app name)
  cmake_parse_arguments(ARG "" "" "SOURCES;REQUIRES;LIBS" ${ARGN})
  foreach(dep IN LISTS ARG_REQUIRES)
    if(NOT TARGET Penner${dep}Lib)
      message(STATUS "Skipping ${name}: needs Penner${dep}Lib")
      return()
    endif()
  endforeach()
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE PennerLib ApplicationUtilLib ${ARG_LIBS})
endfunction()

penner_add_app(generate_field
  SOURCES generate_field.cpp
  REQUIRES Field)

penner_add_app(view_field
  SOURCES view_field.cpp
  REQUIRES Field)

penner_add_app(optimize_metric
  SOURCES optimize_metric.cpp
  REQUIRES Field Optimization)

penner_add_app(parametrize_seamless
  SOURCES parametrize_seamless.cpp
  REQUIRES Holonomy)

penner_add_app(generate_aligned_field
  SOURCES generate_aligned_field.cpp
  REQUIRES Feature)

penner_add_app(parameterize_aligned
  SOURCES parameterize_aligned.cpp
  REQUIRES Feature)

penner_add_app(parametrize_scalable
  SOURCES parametrize_scalable.cpp
  REQUIRES Resolution)

penner_add_app(convert_mesh
  SOURCES convert_mesh.cpp)

penner_add_app(test_interface
  SOURCES test_interface.cpp
  REQUIRES Feature)

# add_executable(optimize_shear
#   optimize_shear.cpp
# )
# target_link_libraries(optimize_shear PRIVATE
#   PennerLib
#   CLI11::CLI11
# )

# only build visualization executables if visualization enabled
if (NOT USE_MULTIPRECISION AND ENABLE_VISUALIZATION)
  penner_add_app(view_quad_mesh
    SOURCES view_quad_mesh.cpp
    REQUIRES Feature)

  penner_add_app(view_seamless
    SOURCES view_seamless.cpp
    REQUIRES Feature)
endif()
