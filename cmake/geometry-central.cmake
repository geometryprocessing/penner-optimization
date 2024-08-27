if(TARGET geometry-central)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    geometry-central
    SYSTEM
    GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
)
FetchContent_MakeAvailable(geometry-central)
