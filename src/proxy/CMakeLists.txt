# openvr_api.dll proxy. Output: build/bin/<Config>/proxy/openvr_api.dll plus openvr_api_orig.dll next to it
# (the official 0.9.12 DLL): exactly the two files that go into the game's x64\ folder.
add_library(qvfr_proxy SHARED
  exports.cpp
  wrappers.cpp
  frame.cpp
  experiment.cpp
  focus_geometry.cpp
  callers.cpp
  hooks.cpp
  probe.cpp
  quad.cpp
  shots.cpp
  modern_vr.cpp
  sysinfo.cpp
  common.cpp
)
target_link_libraries(qvfr_proxy PRIVATE openvr_0912 minhook bcrypt version psapi shell32 ole32 windowscodecs)
target_compile_definitions(qvfr_proxy PRIVATE VR_API_EXPORT)
# /EHa instead of the default /EHsc: an SEH fault caught by Guarded/Safe then unwinds C++ objects (lock_guard),
# so no mutex stays locked after a fault in our code. Directory scope: only this target is built here.
string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
target_compile_options(qvfr_proxy PRIVATE /W4 /EHa)
set_target_properties(qvfr_proxy PROPERTIES
  OUTPUT_NAME openvr_api
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/proxy
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)
add_custom_command(TARGET qvfr_proxy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVR_0912_DLL} $<TARGET_FILE_DIR:qvfr_proxy>/openvr_api_orig.dll
  VERBATIM)
