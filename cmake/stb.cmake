if(TARGET penner_stb)
    return()
endif()

# Header-only stb target for image I/O.
#
# Deliberately not named `stb`: polyscope defines a target with that name.
# No implementation is compiled here; consumers define STB_IMAGE_IMPLEMENTATION /
# STB_IMAGE_WRITE_IMPLEMENTATION together with STB_IMAGE_STATIC /
# STB_IMAGE_WRITE_STATIC in a single source file, so the stbi_* symbols have
# internal linkage and cannot clash with polyscope's copy in the same binary.
include(FetchContent)
FetchContent_Declare(
    stb
    SYSTEM
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f67165c2bb2af3060ecae7d20d6f731173485ad0
)
FetchContent_MakeAvailable(stb)

add_library(penner_stb INTERFACE)
target_include_directories(penner_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})
