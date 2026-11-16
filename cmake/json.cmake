if(TARGET nlohmann_json::nlohmann_json)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    json
    SYSTEM
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    # Must match the json.hpp vendored by seamless-optimization (ExtremeOpt.h
    # includes it directly); otherwise the versioned ABI namespaces differ
    # (json_abi_v3_11_2 vs json_abi_v3_12_0) and functions taking json fail to link.
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

