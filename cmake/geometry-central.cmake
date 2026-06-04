if(TARGET geometry-central)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    geometry-central
    SYSTEM
    GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
    GIT_TAG 52a533bd9d635a3385af3b31f8bab78684d5b20f
)
FetchContent_MakeAvailable(geometry-central)
