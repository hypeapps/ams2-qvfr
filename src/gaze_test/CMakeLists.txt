add_executable(qvfr_gaze_test WIN32 main.cpp)
target_link_libraries(qvfr_gaze_test PRIVATE openvr_sdk d3d11 dxgi d3dcompiler shell32 ole32 windowscodecs)
target_compile_options(qvfr_gaze_test PRIVATE /W4)

# openvr_api.dll is loaded dynamically (LoadLibrary), so both versions sit next to the exe under different names.
add_custom_command(TARGET qvfr_gaze_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVR_SDK_DLL} $<TARGET_FILE_DIR:qvfr_gaze_test>/openvr_api.dll
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENVR_0912_DLL} $<TARGET_FILE_DIR:qvfr_gaze_test>/openvr_api_0912.dll
  VERBATIM)
