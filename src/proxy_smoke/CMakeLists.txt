# Proxy test without the game: loads build/bin/<Config>/proxy/openvr_api.dll the way AMS2 does (VR_Init 0.9.12).
# --xr-layer and --openxr-path test the OpenXR path (xr_smoke.cpp).
# --shots, --quad-compose, --focus-geometry, --focus-cull and --clones test the proxy's shot, composition, focus geometry,
# focus culling and clones modules directly, so their sources are compiled in here as well.
add_executable(qvfr_proxy_smoke main.cpp xr_smoke.cpp sun_fx_smoke.cpp mvr_vs_smoke.cpp mvr_ps_smoke.cpp mvr_pass_smoke.cpp mvr_forward_smoke.cpp taa_fix_smoke.cpp ../proxy/shots.cpp ../proxy/compose.cpp ../proxy/focus_geometry.cpp
               ../proxy/focus_cull.cpp ../proxy/common.cpp ../proxy/clones.cpp ../proxy/hooks.cpp ../proxy/sysinfo.cpp ../proxy/diag.cpp
               ../proxy/sun_fx.cpp ../proxy/mvr_vs.cpp ../proxy/mvr_ps.cpp ../proxy/mvr_pass.cpp ../proxy/taa_views.cpp ../proxy/gpu.cpp)
target_include_directories(qvfr_proxy_smoke PRIVATE ../proxy)
target_link_libraries(qvfr_proxy_smoke PRIVATE qvfr_shaders qvfr_dxbc nvapi openvr_0912 openxr_headers minhook d3d11 d3dcompiler windowscodecs ole32 shell32
                      bcrypt version psapi)
target_compile_options(qvfr_proxy_smoke PRIVATE /W4)
add_dependencies(qvfr_proxy_smoke qvfr_proxy)
