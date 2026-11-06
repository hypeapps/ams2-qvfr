# OpenVR SDK 2.15.6: headers for the tools (gaze test) and for newer interfaces (IVRSystem_026, IVRInput_011).
add_library(openvr_sdk INTERFACE)
target_include_directories(openvr_sdk INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/openvr-2.15.6/headers)
set(OPENVR_SDK_DLL ${CMAKE_CURRENT_SOURCE_DIR}/openvr-2.15.6/bin/win64/openvr_api.dll CACHE INTERNAL "")

# OpenVR 0.9.12: the version AMS2 talks to (IVRSystem_009, IVRCompositor_009, IVROverlay_007).
# The DLL is byte-identical to the game's x64\openvr_api.dll (MD5 864be5a92276ce2b902b6f3a0f850703).
add_library(openvr_0912 INTERFACE)
target_include_directories(openvr_0912 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/openvr-0.9.12/headers)
set(OPENVR_0912_DLL ${CMAKE_CURRENT_SOURCE_DIR}/openvr-0.9.12/bin/win64/openvr_api.dll CACHE INTERNAL "")
set(OPENCOMPOSITE_DLL ${CMAKE_CURRENT_SOURCE_DIR}/opencomposite-1.0.1539/bin/win64/openvr_api.dll CACHE INTERNAL "")

# MinHook 1.3.4 (BSD-2-Clause), built from source as a static library.
add_library(minhook STATIC
  minhook/src/buffer.c
  minhook/src/hook.c
  minhook/src/trampoline.c
  minhook/src/hde/hde64.c
)
target_include_directories(minhook PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/minhook/include)
target_compile_options(minhook PRIVATE /W3)

# OpenXR SDK 1.1.65: headers only, for the OpenXR API layer in the proxy (path=openxr, M8).
add_library(openxr_headers INTERFACE)
target_include_directories(openxr_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/openxr-1.1.65/include)

# FreeType 2.15.0 (FreeType License, FTL) and RmlUi 6.3 (MIT): the settings GUI (src/config). Both static, FreeType
# without its optional compression and shaping libraries.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(opt FT_DISABLE_ZLIB FT_DISABLE_BZIP2 FT_DISABLE_PNG FT_DISABLE_HARFBUZZ FT_DISABLE_BROTLI)
  set(${opt} ON CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(freetype-2.15.0 EXCLUDE_FROM_ALL)
add_library(Freetype::Freetype ALIAS freetype)
target_compile_options(freetype PRIVATE /wd4005)  # builds/windows/ftsystem.c defines WIN32_LEAN_AND_MEAN again
set(RMLUI_FONT_ENGINE "freetype" CACHE STRING "" FORCE)
set(RMLUI_THIRDPARTY_CONTAINERS OFF CACHE BOOL "" FORCE)  # std containers: a settings window does not need the faster ones
add_subdirectory(rmlui-6.3 EXCLUDE_FROM_ALL)

# NVIDIA NVAPI R615 (MIT): headers and the static nvapi64.lib, for the Multi-View Rendering test (src/mvr_test)
# and the proxy's MVR vertex shader patch (src/proxy/mvr_vs.cpp; nvapi64.dll is loaded at run time).
add_library(nvapi INTERFACE)
target_include_directories(nvapi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nvapi-r615)
target_link_libraries(nvapi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nvapi-r615/amd64/nvapi64.lib)
