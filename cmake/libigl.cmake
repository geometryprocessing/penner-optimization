if(TARGET igl::core)
  return()
endif()

include(FetchContent)
  FetchContent_Declare(
  libigl
  SYSTEM
  GIT_REPOSITORY https://github.com/rjc8237/libigl.git
  GIT_TAG penner
)
FetchContent_MakeAvailable(libigl)
