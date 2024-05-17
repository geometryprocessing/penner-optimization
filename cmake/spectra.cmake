if(TARGET Spectra)
  return()
endif()

include(FetchContent)
  FetchContent_Declare(
  spectra
  SYSTEM
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
)
FetchContent_MakeAvailable(spectra)
