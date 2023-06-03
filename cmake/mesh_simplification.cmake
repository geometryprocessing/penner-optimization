include(FetchContent)
include(ExternalProject)
#FetchContent_Declare(
#    mesh_simplification
#    GIT_REPOSITORY https://github.com/rjc8237/mesh_simplification.git
#)
ExternalProject_Add(mesh_simplification
    GIT_REPOSITORY https://github.com/rjc8237/mesh_simplification.git
    #GIT_TAG main
    GIT_TAG 45d4b7bd657d8e70bcdda077a03fe5738bc9d744
    BINARY_DIR ${PROJECT_SOURCE_DIR}/ext
    INSTALL_COMMAND ""
    BUILD_ALWAYS FALSE
    UPDATE_COMMAND ""
)
