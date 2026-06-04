if(TARGET nlohmann_json::nlohmann_json)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    json
    SYSTEM
    GIT_REPOSITORY https://github.com/nlohmann/json.git
)
FetchContent_MakeAvailable(json)

