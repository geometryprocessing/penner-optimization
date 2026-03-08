include(FetchContent)
FetchContent_Declare(
    symmetric-dirichlet
    SYSTEM
    GIT_REPOSITORY https://github.com/rjc8237/symmetric-dirichlet.git
    GIT_TAG energy_dev
)
FetchContent_MakeAvailable(symmetric-dirichlet)

