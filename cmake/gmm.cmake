include(FetchContent)

FetchContent_Declare(
    gmm
    SYSTEM
    GIT_REPOSITORY https://github.com/rjc8237/CoMISo.git
    GIT_TAG        master   # or a specific commit hash
)
FetchContent_MakeAvailable(gmm)

#  FetchContent_Declare(
#      gmm_src
#      GIT_REPOSITORY https://git.savannah.gnu.org/git/getfem.git
#      GIT_TAG        master   # or a specific commit hash
#  )
# FetchContent_Populate(gmm_src)

# # Generate gmm_arch_config.h into your build tree
# set(GMM_GEN_DIR "${CMAKE_BINARY_DIR}/_gmm_generated")
# file(MAKE_DIRECTORY "${GMM_GEN_DIR}/gmm")

# configure_file(
#   "${gmm_src_SOURCE_DIR}/cmake/gmm_arch_config.h.in"
#   "${GMM_GEN_DIR}/gmm/gmm_arch_config.h"
#   @ONLY
# )

# add_library(gmm INTERFACE)
# target_include_directories(gmm INTERFACE
#   "${gmm_src_SOURCE_DIR}/src"   # provides gmm/gmm.h etc
#   "${GMM_GEN_DIR}"              # provides gmm/gmm_arch_config.h
# )