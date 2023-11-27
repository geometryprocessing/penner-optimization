if(TARGET myles_data)
	return()
endif()

include(ExternalProject)
include(FetchContent)

set(MYLES_DATA_ROOT "${PROJECT_SOURCE_DIR}/data/" CACHE PATH "Where should the project download and look for test data?")

# ExternalProject_Add(
# wmtk_data_download
# PREFIX "${FETCHCONTENT_BASE_DIR}/wmtk-test-data"
# SOURCE_DIR ${WMTK_DATA_ROOT}
#
# GIT_REPOSITORY https://github.com/wildmeshing/data.git
# GIT_TAG 6cfdde9b0928c6d8f1ade9890f1e128a3fd48e44
#
# CONFIGURE_COMMAND ""
# BUILD_COMMAND ""
# INSTALL_COMMAND ""
# LOG_DOWNLOAD ON
# )
add_library(myles_data INTERFACE)

# add_dependencies(wmtk_data wmtk_data_download)
target_compile_definitions(myles_data INTERFACE MYLES_DATA_DIR=\"${WMTK_DATA_ROOT}\")