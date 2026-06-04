if (TARGET Catch2::Catch2WithMain)
  return()
endif()

Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.15.0
)
FetchContent_MakeAvailable(Catch2)
