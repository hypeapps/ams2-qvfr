# DXBC container helpers and the Multi-View Rendering vertex shader patch.
add_library(qvfr_dxbc STATIC dxbc.cpp mvr_patch.cpp)
target_include_directories(qvfr_dxbc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(qvfr_dxbc PRIVATE /W4)
