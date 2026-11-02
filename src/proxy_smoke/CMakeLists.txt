# Proxy test without the game: loads build/bin/<Config>/proxy/openvr_api.dll the way AMS2 does (VR_Init 0.9.12).
# --shots, --quad-compose and --focus-geometry test the proxy's shot, composition and focus geometry modules directly,
# so their sources are compiled in here as well.
add_executable(qvfr_proxy_smoke main.cpp ../proxy/shots.cpp ../proxy/compose.cpp ../proxy/focus_geometry.cpp
               ../proxy/common.cpp)
target_include_directories(qvfr_proxy_smoke PRIVATE ../proxy)
target_link_libraries(qvfr_proxy_smoke PRIVATE qvfr_shaders openvr_0912 d3d11 windowscodecs ole32 shell32)
target_compile_options(qvfr_proxy_smoke PRIVATE /W4)
add_dependencies(qvfr_proxy_smoke qvfr_proxy)
