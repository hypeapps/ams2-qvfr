# openvr_api.dll proxy. Output: build/bin/<Config>/proxy/openvr_api.dll plus openvr_api_orig.dll next to it
# (the official 0.9.12 DLL): exactly the two files that go into the game's x64\ folder.
add_library(qvfr_proxy SHARED
  exports.cpp
  wrappers.cpp
  frame.cpp
  experiment.cpp
  modern_vr.cpp
  sysinfo.cpp
  common.cpp
)
target_link_libraries(qvfr_proxy PRIVATE openvr_0912 bcrypt version psapi shell32 ole32)
target_compile_definitions(qvfr_proxy PRIVATE VR_API_EXPORT)
target_compile_options(qvfr_proxy PRIVATE /W4)
set_target_properties(qvfr_proxy PROPERTIES
  OUTPUT_NAME openvr_api
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/proxy
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)
add_custom_command(TARGET qvfr_proxy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVR_0912_DLL} $<TARGET_FILE_DIR:qvfr_proxy>/openvr_api_orig.dll
  VERBATIM)
