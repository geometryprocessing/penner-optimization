if(TARGET CoMISo)
  return()
endif()

FetchContent_Declare(
    comiso
    SYSTEM
    GIT_REPOSITORY https://github.com/rjc8237/CoMISo.git
    GIT_TAG        master   # or a specific commit hash
)
FetchContent_MakeAvailable(comiso)

add_library(CoMISo::CoMISo ALIAS CoMISo)

# Copy .hh headers into a subfolder `CoMISo/`
file(GLOB_RECURSE INC_FILES "${comiso_SOURCE_DIR}/*.hh" "${comiso_SOURCE_DIR}/*.cc")
set(output_folder "${CMAKE_CURRENT_BINARY_DIR}/CoMISo/include/CoMISo")
message(VERBOSE "Copying CoMISo headers to '${output_folder}'")
foreach(filepath IN ITEMS ${INC_FILES})
    file(RELATIVE_PATH filename "${comiso_SOURCE_DIR}" ${filepath})
    configure_file(${filepath} "${output_folder}/${filename}" COPYONLY)
endforeach()

target_include_directories(CoMISo PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/CoMISo/include)

set_target_properties(CoMISo PROPERTIES FOLDER ThirdParty)

# include(FetchContent)
#   FetchContent_Declare(
#   comiso
#   SYSTEM
#   GIT_REPOSITORY https://gitlab.inf.unibe.ch/CGG-public/CoMISo/CoMISo.git
# )
# FetchContent_MakeAvailable(comiso)
