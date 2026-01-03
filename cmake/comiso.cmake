if(TARGET CoMISo)
  return()
endif()

include(FetchContent)
  FetchContent_Declare(
  comiso
  SYSTEM
  GIT_REPOSITORY https://gitlab.inf.unibe.ch/CGG-public/CoMISo/CoMISo.git
)
FetchContent_MakeAvailable(comiso)
