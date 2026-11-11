# NR prototype (dlaa-plan D7): nvngx.dll_qvfr_nr.dll next to the proxy's files in qvfr/. Its name must contain
# "nvngx.dll" (the model's caller check); it forwards the proxy's calls to the user's nvngx_dlssnr.dll.
add_library(qvfr_nr_forwarder SHARED forwarder.cpp)
target_compile_options(qvfr_nr_forwarder PRIVATE /W4)
set_target_properties(qvfr_nr_forwarder PROPERTIES
  OUTPUT_NAME nvngx.dll_qvfr_nr
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/proxy/qvfr
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)
