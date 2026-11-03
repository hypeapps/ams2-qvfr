# openvr_api.dll proxy. Output: build/bin/<Config>/proxy/ in the game's x64 layout: openvr_api.dll (proxy),
# qvfr/openvr/openvr_api.dll (the official 0.9.12 DLL) and qvfr/XR_APILAYER_AMS2QVFR_eye_gaze.json (OpenXR layer manifest).
# Composition shaders (M4.4): compiled with fxc from the Windows SDK into headers with the bytecode.
find_program(QVFR_FXC fxc HINTS "$ENV{WindowsSdkVerBinPath}x64" REQUIRED)
set(QVFR_SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(QVFR_COMPOSE_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compose.hlsl)
add_custom_command(
  OUTPUT ${QVFR_SHADER_OUT}/compose_vs.h ${QVFR_SHADER_OUT}/compose_ps.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QVFR_SHADER_OUT}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T vs_5_0 /E VSMain /Vn kComposeVS
          /Fh ${QVFR_SHADER_OUT}/compose_vs.h ${QVFR_COMPOSE_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSMain /Vn kComposePS
          /Fh ${QVFR_SHADER_OUT}/compose_ps.h ${QVFR_COMPOSE_HLSL}
  DEPENDS ${QVFR_COMPOSE_HLSL}
  COMMENT "fxc compose.hlsl"
  VERBATIM)
# Linking this target gives the include directory and builds the headers first (proxy and smoke test).
add_library(qvfr_shaders INTERFACE ${QVFR_SHADER_OUT}/compose_vs.h ${QVFR_SHADER_OUT}/compose_ps.h)
target_include_directories(qvfr_shaders INTERFACE ${QVFR_SHADER_OUT})

enable_language(ASM_MASM)  # export thunks (generated/thunks.asm)

add_library(qvfr_proxy SHARED
  exports.cpp
  generated/thunks.asm
  wrappers.cpp
  frame.cpp
  experiment.cpp
  focus_geometry.cpp
  compose.cpp
  gpu.cpp
  callers.cpp
  clones.cpp
  hooks.cpp
  probe.cpp
  quad.cpp
  shots.cpp
  modern_vr.cpp
  runtime.cpp
  diag.cpp
  xr_layer.cpp
  sysinfo.cpp
  common.cpp
)
target_link_libraries(qvfr_proxy PRIVATE qvfr_shaders openvr_0912 openxr_headers minhook bcrypt version psapi shell32 ole32 windowscodecs)
target_compile_definitions(qvfr_proxy PRIVATE VR_API_EXPORT)
# /EHa instead of the default /EHsc: an SEH fault caught by Guarded/Safe then unwinds C++ objects (lock_guard),
# so no mutex stays locked after a fault in our code. Directory scope: only this target is built here.
string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_compile_options(qvfr_proxy PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4;/EHa>")
set_target_properties(qvfr_proxy PROPERTIES
  OUTPUT_NAME openvr_api
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/proxy
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)
add_custom_command(TARGET qvfr_proxy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVR_0912_DLL} $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/openvr/openvr_api.dll
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/XR_APILAYER_AMS2QVFR_eye_gaze.json
          $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/XR_APILAYER_AMS2QVFR_eye_gaze.json
  # Spare copy for the settings GUI's Repair (after Steam's file verification restored the Valve DLL).
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:qvfr_proxy> $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/proxy/openvr_api.dll
  VERBATIM)
