include(FetchContent)
FetchContent_Declare(
    conformal_ideal_delaunay
    SYSTEM
    GIT_REPOSITORY https://github.com/rjc8237/ConformalIdealDelaunay.git
    GIT_TAG d139f38edfac4395951f38dd4a0c5b9cda1d79e3
)
FetchContent_MakeAvailable(conformal_ideal_delaunay)
