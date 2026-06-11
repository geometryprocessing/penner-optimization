set(FieldSrc
  field.cpp
  frame_field.cpp
  cross_field.cpp
  facet_field.cpp
  forms.cpp
  intrinsic_field.cpp
)

add_library(PennerFieldLib
  ${FieldSrc}
)
target_include_directories(PennerFieldLib PUBLIC ../../include/field)
target_link_libraries(PennerFieldLib PUBLIC
  PennerCoreLib
  ${VISUALIZATION_LIBS}
)
target_link_libraries(PennerFieldLib PRIVATE
  geometry-central
  nlohmann_json::nlohmann_json
  DirectionalLib
  ${COMISO_LIBS}
)
target_compile_definitions(PennerFieldLib PUBLIC
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)
if ((NOT USE_MULTIPRECISION) AND (USE_WARNINGS))
    target_compile_options(PennerFieldLib PRIVATE
      -Wall -Wpedantic -Wextra -Werror
    )
endif()
