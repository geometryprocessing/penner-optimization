 message("Checking for pybind11")
if(TARGET pybind11::module)
  return()
endif()

 message("Fetching pybind11")
include(FetchContent)
FetchContent_Declare(
  pybind11
  SYSTEM
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  message("Adding pybind11 source")
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
  include_directories(${pybind11_SOURCE_DIR}/include)
endif()
