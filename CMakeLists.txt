cmake_minimum_required(VERSION 3.28)
project(ams2_qvfr VERSION 0.2.0 LANGUAGES C CXX RC)

if(NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "This project builds with MSVC x64 only (AMS2 is 64-bit).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static CRT: a proxy loaded into the game must not depend on the user's vcruntime version.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8;/permissive->" /Zi)
add_link_options(/DEBUG:FULL)

# All binaries go to build/bin/<Config>.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(third_party)
add_subdirectory(src/gaze_test)
add_subdirectory(src/proxy)
add_subdirectory(src/proxy_smoke)
add_subdirectory(src/config)
