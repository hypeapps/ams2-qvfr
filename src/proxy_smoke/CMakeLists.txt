# Proxy test without the game: loads build/bin/<Config>/proxy/openvr_api.dll the way AMS2 does (VR_Init 0.9.12).
# --xr-layer and --openxr-path test the OpenXR path (xr_smoke.cpp).
# --shots, --quad-compose, --focus-geometry, --focus-cull and --clones test the proxy's shot, composition, focus geometry,
# focus culling and clones modules directly, so their sources are compiled in here as well.
add_executable(qvfr_proxy_smoke main.cpp xr_smoke.cpp sun_fx_smoke.cpp mvr_vs_smoke.cpp mvr_ps_smoke.cpp mvr_pass_smoke.cpp mvr_forward_smoke.cpp mvr_motion_smoke.cpp taa_fix_smoke.cpp nr_bridge_smoke.cpp ../proxy/nr_bridge.cpp ../proxy/shots.cpp ../proxy/compose.cpp ../proxy/focus_geometry.cpp
               ../proxy/focus_cull.cpp ../proxy/fwd_rec.cpp ../proxy/common.cpp ../proxy/clones.cpp ../proxy/hooks.cpp ../proxy/hook_prof.cpp ../proxy/sysinfo.cpp ../proxy/diag.cpp
               ../proxy/sun_fx.cpp ../proxy/mvr_vs.cpp ../proxy/mvr_ps.cpp ../proxy/mvr_pass.cpp ../proxy/mvr_motion.cpp ../proxy/taa_views.cpp ../proxy/gpu.cpp
               panel_smoke.cpp ../proxy/panel_ui.cpp ${PROJECT_SOURCE_DIR}/third_party/rmlui-6.3/Backends/RmlUi_Renderer_DX11.cpp
               ../proxy/panel.rc ../proxy/fps_hud.cpp)
target_include_directories(qvfr_proxy_smoke PRIVATE ../proxy)
target_link_libraries(qvfr_proxy_smoke PRIVATE qvfr_shaders qvfr_dxbc nvapi openvr_0912 openxr_headers minhook d3d11 dxgi d3dcompiler windowscodecs ole32 shell32
                      bcrypt version psapi RmlUi::Core gdi32)
target_include_directories(qvfr_proxy_smoke PRIVATE ${PROJECT_SOURCE_DIR}/third_party/rmlui-6.3/Backends)
set_source_files_properties(${PROJECT_SOURCE_DIR}/third_party/rmlui-6.3/Backends/RmlUi_Renderer_DX11.cpp PROPERTIES
                            COMPILE_OPTIONS "/wd4005")
target_compile_options(qvfr_proxy_smoke PRIVATE /W4)
add_dependencies(qvfr_proxy_smoke qvfr_proxy)
# The NR bridge's NGX code needs only the DLSS SDK's headers (it calls the driver's core through GetProcAddress).
set(QVFR_SMOKE_DLSS_SDK ${PROJECT_SOURCE_DIR}/third_party/dlss-310.9.16)
if(EXISTS ${QVFR_SMOKE_DLSS_SDK}/include/nvsdk_ngx.h)
  set_source_files_properties(../proxy/nr_bridge.cpp PROPERTIES COMPILE_DEFINITIONS QVFR_HAVE_NGX_HEADERS)
  target_include_directories(qvfr_proxy_smoke SYSTEM PRIVATE ${QVFR_SMOKE_DLSS_SDK}/include)
endif()
