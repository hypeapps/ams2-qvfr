# qvfr_config.exe: the settings GUI (M8.6). RmlUi with its Win32 + D3D11 backend, static CRT, no runtime to install;
# the page files and fonts are resources (qvfr_config.rc). Shares presets.h and default_ini.inc with the proxy. The
# post-build copy puts it into the game layout next to the proxy build (proxy/qvfr/qvfr_config.exe).
set(RML_BACKENDS ${CMAKE_SOURCE_DIR}/third_party/rmlui-6.3/Backends)
add_executable(qvfr_config WIN32
  main.cpp settings.cpp ini_file.cpp log_reader.cpp files.cpp selftest.cpp
  ${RML_BACKENDS}/RmlUi_Platform_Win32.cpp
  ${RML_BACKENDS}/RmlUi_Renderer_DX11.cpp
  ${RML_BACKENDS}/RmlUi_Backend_Win32_DX11.cpp
  qvfr_config.rc)
target_include_directories(qvfr_config PRIVATE ../proxy ${RML_BACKENDS})
target_link_libraries(qvfr_config PRIVATE RmlUi::Core d3d11 dxgi d3dcompiler dwmapi bcrypt shell32)
target_compile_definitions(qvfr_config PRIVATE QVFR_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  QVFR_VERSION_MINOR=${PROJECT_VERSION_MINOR} QVFR_VERSION_PATCH=${PROJECT_VERSION_PATCH})
foreach(src main.cpp settings.cpp ini_file.cpp log_reader.cpp files.cpp selftest.cpp)
  set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS /W4)
endforeach()
# The backend's own Windows header defines UNICODE and WIN32_LEAN_AND_MEAN again (C4005).
set_source_files_properties(${RML_BACKENDS}/RmlUi_Platform_Win32.cpp ${RML_BACKENDS}/RmlUi_Renderer_DX11.cpp
  ${RML_BACKENDS}/RmlUi_Backend_Win32_DX11.cpp PROPERTIES COMPILE_OPTIONS /wd4005)
file(GLOB QVFR_CONFIG_FONTS "${PROJECT_SOURCE_DIR}/third_party/fonts/*/*.ttf")
set_source_files_properties(qvfr_config.rc PROPERTIES OBJECT_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/ui/main.rml;${CMAKE_CURRENT_SOURCE_DIR}/ui/style.rcss;${CMAKE_CURRENT_SOURCE_DIR}/qvfr_config.ico;${QVFR_CONFIG_FONTS}")
add_custom_command(TARGET qvfr_config POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:qvfr_config>/proxy/qvfr
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:qvfr_config> $<TARGET_FILE_DIR:qvfr_config>/proxy/qvfr/qvfr_config.exe)
