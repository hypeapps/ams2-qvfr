# Proxy test without the game: loads build/bin/<Config>/proxy/openvr_api.dll the way AMS2 does (VR_Init 0.9.12).
add_executable(qvfr_proxy_smoke main.cpp)
target_link_libraries(qvfr_proxy_smoke PRIVATE openvr_0912 d3d11)
target_compile_options(qvfr_proxy_smoke PRIVATE /W4)
add_dependencies(qvfr_proxy_smoke qvfr_proxy)
