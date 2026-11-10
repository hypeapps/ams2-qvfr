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
set(QVFR_SUNFX_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sun_fx.hlsl)
add_custom_command(
  OUTPUT ${QVFR_SHADER_OUT}/sun_fx_vs.h ${QVFR_SHADER_OUT}/sun_fx_ps.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QVFR_SHADER_OUT}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T vs_5_0 /E VSMain /Vn kSunFxVS
          /Fh ${QVFR_SHADER_OUT}/sun_fx_vs.h ${QVFR_SUNFX_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSMain /Vn kSunFxPS
          /Fh ${QVFR_SHADER_OUT}/sun_fx_ps.h ${QVFR_SUNFX_HLSL}
  DEPENDS ${QVFR_SUNFX_HLSL}
  COMMENT "fxc sun_fx.hlsl"
  VERBATIM)
set(QVFR_DLAA_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/dlaa.hlsl)
add_custom_command(
  OUTPUT ${QVFR_SHADER_OUT}/dlaa_mask_cs.h ${QVFR_SHADER_OUT}/dlaa_cockpit_cs.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QVFR_SHADER_OUT}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T cs_5_0 /E CSMask /Vn kDlaaMaskCS
          /Fh ${QVFR_SHADER_OUT}/dlaa_mask_cs.h ${QVFR_DLAA_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T cs_5_0 /E CSCockpit /Vn kDlaaCockpitCS
          /Fh ${QVFR_SHADER_OUT}/dlaa_cockpit_cs.h ${QVFR_DLAA_HLSL}
  DEPENDS ${QVFR_DLAA_HLSL}
  COMMENT "fxc dlaa.hlsl"
  VERBATIM)
set(QVFR_SR_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sr.hlsl)
add_custom_command(
  OUTPUT ${QVFR_SHADER_OUT}/sr_vs.h ${QVFR_SHADER_OUT}/sr_color_ps.h ${QVFR_SHADER_OUT}/sr_depth_ps.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QVFR_SHADER_OUT}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T vs_5_0 /E VSFull /Vn kSrVS
          /Fh ${QVFR_SHADER_OUT}/sr_vs.h ${QVFR_SR_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSColor /Vn kSrColorPS
          /Fh ${QVFR_SHADER_OUT}/sr_color_ps.h ${QVFR_SR_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSDepth /Vn kSrDepthPS
          /Fh ${QVFR_SHADER_OUT}/sr_depth_ps.h ${QVFR_SR_HLSL}
  DEPENDS ${QVFR_SR_HLSL}
  COMMENT "fxc sr.hlsl"
  VERBATIM)
set(QVFR_MVRPASS_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mvr_pass.hlsl)
add_custom_command(
  OUTPUT ${QVFR_SHADER_OUT}/mvr_pass_vs.h ${QVFR_SHADER_OUT}/mvr_pass_depth_ps.h ${QVFR_SHADER_OUT}/mvr_pass_mask_ps.h
         ${QVFR_SHADER_OUT}/mvr_pass_depth_in_ps.h ${QVFR_SHADER_OUT}/mvr_pass_blit_ps.h
         ${QVFR_SHADER_OUT}/mvr_pass_color_in_ms_ps.h ${QVFR_SHADER_OUT}/mvr_pass_color_out_ms_ps.h
         ${QVFR_SHADER_OUT}/mvr_pass_depth_in_ms_ps.h ${QVFR_SHADER_OUT}/mvr_pass_depth_ms_ps.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QVFR_SHADER_OUT}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T vs_5_0 /E VSMain /Vn kMvrPassVS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_vs.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSDepth /Vn kMvrPassDepthPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_depth_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSMask /Vn kMvrPassMaskPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_mask_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSDepthIn /Vn kMvrPassDepthInPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_depth_in_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSBlit /Vn kMvrPassBlitPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_blit_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSColorInMs /Vn kMvrPassColorInMsPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_color_in_ms_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSColorOutMs /Vn kMvrPassColorOutMsPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_color_out_ms_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSDepthInMs /Vn kMvrPassDepthInMsPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_depth_in_ms_ps.h ${QVFR_MVRPASS_HLSL}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T ps_5_0 /E PSDepthMs /Vn kMvrPassDepthMsPS
          /Fh ${QVFR_SHADER_OUT}/mvr_pass_depth_ms_ps.h ${QVFR_MVRPASS_HLSL}
  DEPENDS ${QVFR_MVRPASS_HLSL}
  COMMENT "fxc mvr_pass.hlsl"
  VERBATIM)
set(QVFR_MVRMOTION_HLSL ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mvr_motion.hlsl)
add_custom_command(
  OUTPUT ${QVFR_SHADER_OUT}/mvr_motion_cs.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${QVFR_SHADER_OUT}
  COMMAND ${QVFR_FXC} /nologo /O3 /Qstrip_reflect /Qstrip_debug /T cs_5_0 /E CSFocusMotion /Vn kMvrMotionCS
          /Fh ${QVFR_SHADER_OUT}/mvr_motion_cs.h ${QVFR_MVRMOTION_HLSL}
  DEPENDS ${QVFR_MVRMOTION_HLSL}
  COMMENT "fxc mvr_motion.hlsl"
  VERBATIM)
# Linking this target gives the include directory and builds the headers first (proxy and smoke test).
add_library(qvfr_shaders INTERFACE ${QVFR_SHADER_OUT}/compose_vs.h ${QVFR_SHADER_OUT}/compose_ps.h
            ${QVFR_SHADER_OUT}/sun_fx_vs.h ${QVFR_SHADER_OUT}/sun_fx_ps.h ${QVFR_SHADER_OUT}/dlaa_mask_cs.h
            ${QVFR_SHADER_OUT}/dlaa_cockpit_cs.h ${QVFR_SHADER_OUT}/mvr_pass_vs.h ${QVFR_SHADER_OUT}/mvr_pass_depth_ps.h
            ${QVFR_SHADER_OUT}/mvr_pass_mask_ps.h ${QVFR_SHADER_OUT}/mvr_pass_depth_in_ps.h
            ${QVFR_SHADER_OUT}/mvr_pass_blit_ps.h ${QVFR_SHADER_OUT}/mvr_pass_color_in_ms_ps.h
            ${QVFR_SHADER_OUT}/mvr_pass_color_out_ms_ps.h ${QVFR_SHADER_OUT}/mvr_pass_depth_in_ms_ps.h
            ${QVFR_SHADER_OUT}/mvr_pass_depth_ms_ps.h ${QVFR_SHADER_OUT}/mvr_motion_cs.h
            ${QVFR_SHADER_OUT}/sr_vs.h ${QVFR_SHADER_OUT}/sr_color_ps.h ${QVFR_SHADER_OUT}/sr_depth_ps.h)
target_include_directories(qvfr_shaders INTERFACE ${QVFR_SHADER_OUT})

enable_language(ASM_MASM)  # export thunks (generated/thunks.asm)

add_library(qvfr_proxy SHARED
  exports.cpp
  generated/thunks.asm
  wrappers.cpp
  frame.cpp
  experiment.cpp
  focus_geometry.cpp
  focus_cull.cpp
  fwd_rec.cpp
  compose.cpp
  gpu.cpp
  callers.cpp
  clones.cpp
  hooks.cpp
  hook_prof.cpp
  probe.cpp
  quad.cpp
  shots.cpp
  modern_vr.cpp
  runtime.cpp
  diag.cpp
  dlaa.cpp
  sr.cpp
  xr_layer.cpp
  sysinfo.cpp
  sun_fx.cpp
  mvr_ps.cpp
  mvr_vs.cpp
  mvr_pass.cpp
  mvr_motion.cpp
  taa_views.cpp
  common.cpp
)
target_link_libraries(qvfr_proxy PRIVATE qvfr_shaders qvfr_dxbc nvapi openvr_0912 openxr_headers minhook bcrypt version psapi shell32 ole32 windowscodecs)
target_compile_definitions(qvfr_proxy PRIVATE VR_API_EXPORT)
# /EHa instead of the default /EHsc: an SEH fault caught by Guarded/Safe then unwinds C++ objects (lock_guard),
# so no mutex stays locked after a fault in our code. Directory scope: only this target is built here.
string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_compile_options(qvfr_proxy PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4;/EHa>")
# DLAA prototype: the NVIDIA DLSS SDK is optional and stays out of git (its license allows distribution only as object
# code), so the proxy builds DLSS in only when the SDK is unpacked locally in third_party/.
set(QVFR_DLSS_SDK ${PROJECT_SOURCE_DIR}/third_party/dlss-310.9.16)
if(EXISTS ${QVFR_DLSS_SDK}/include/nvsdk_ngx.h AND EXISTS ${QVFR_DLSS_SDK}/lib/nvsdk_ngx_s.lib)
  message(STATUS "DLSS SDK found (${QVFR_DLSS_SDK}): DLAA built into the proxy")
  target_compile_definitions(qvfr_proxy PRIVATE QVFR_HAVE_DLSS)
  target_include_directories(qvfr_proxy SYSTEM PRIVATE ${QVFR_DLSS_SDK}/include)
  target_link_libraries(qvfr_proxy PRIVATE
    $<IF:$<CONFIG:Debug>,${QVFR_DLSS_SDK}/lib/nvsdk_ngx_s_dbg.lib,${QVFR_DLSS_SDK}/lib/nvsdk_ngx_s.lib>)
  add_custom_command(TARGET qvfr_proxy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QVFR_DLSS_SDK}/bin/nvngx_dlss.dll
            $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/nvngx_dlss.dll
    VERBATIM)
else()
  message(STATUS "DLSS SDK not found in ${QVFR_DLSS_SDK}: the proxy builds without DLAA")
endif()
set_target_properties(qvfr_proxy PROPERTIES
  OUTPUT_NAME openvr_api
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/proxy
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)
add_custom_command(TARGET qvfr_proxy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVR_0912_DLL} $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/openvr/openvr_api.dll
  # The pinned OpenComposite, the backend of [runtime] path=openxr (GPLv3, shipped unmodified with its licenses).
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCOMPOSITE_DLL} $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/opencomposite/openvr_api.dll
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/XR_APILAYER_AMS2QVFR_eye_gaze.json
          $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/XR_APILAYER_AMS2QVFR_eye_gaze.json
  # Spare copy for the settings GUI's Repair (after Steam's file verification restored the Valve DLL).
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:qvfr_proxy> $<TARGET_FILE_DIR:qvfr_proxy>/qvfr/proxy/openvr_api.dll
  VERBATIM)
