if(TARGET geometry-central)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    geometry-central
    SYSTEM
    GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
    GIT_TAG 03f198748d263cea3faa040c8a399d59c38d67bc
)
FetchContent_MakeAvailable(geometry-central)
