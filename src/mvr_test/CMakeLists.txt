# B1 test of NVIDIA Multi-View Rendering in D3D11 (no game, no headset): correctness of the view routing and CPU/GPU cost.
add_executable(qvfr_mvr_test main.cpp)
target_link_libraries(qvfr_mvr_test PRIVATE nvapi d3d11 dxgi d3dcompiler winmm)
target_compile_options(qvfr_mvr_test PRIVATE /W4)
