# Tests of NVIDIA Multi-View Rendering in D3D11 (no game, no headset): view routing, CPU/GPU cost (B1) and the
# DXBC vertex shader patch (B4) and pixel shader patch (B9b).
add_executable(qvfr_mvr_test main.cpp ps_patch_test.cpp)
target_link_libraries(qvfr_mvr_test PRIVATE qvfr_dxbc nvapi d3d11 dxgi d3dcompiler winmm)
target_compile_options(qvfr_mvr_test PRIVATE /W4)
