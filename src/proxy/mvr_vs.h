// Vertex shaders of the game made ready for NVIDIA Multi-View Rendering (docs/mvr.md, gate B6).
//
// A hook on the device's CreateVertexShader patches every vertex shader the game creates (qvfr_dxbc PatchVsForMvr:
// one extra view with its position and clip distances, matrices in VS constant buffer slot 3) and creates it through
// NvAPI_D3D11_CreateVertexShaderEx with the NV_POSITION and NV_CLIP_DISTANCE_0 semantics. Multi-View Rendering itself
// stays off, so the patched shaders draw view 0 only, as before (B1 case 0, B4 case 13). A refused patch or an NVAPI
// error gives the game the original shader and a log entry with the reason. Only devices for which
// NvAPI_D3D_QueryMultiViewSupport reports Multi-View support get patched shaders.
//
// [debug] mvr_vs_patch=1 turns it on (default off; a change needs a game restart). The hook is installed at VR_Init in
// quad mode, through a temporary WARP device, like the sun effects module.
#pragma once

#include <d3d11.h>

#include <string>

namespace qvfr::mvrvs {

// Constant buffer slot of the view matrices (free in every scene VS of the capture, docs/mvr.md B2).
constexpr unsigned kCbSlot = 3;

// VR_Init in quad mode with the setting on, once per process: NVAPI and the D3D11 hook. False: the module stays off.
bool Install();
// Render thread, every frame start: `ctx` = the game's immediate context (null before the first Submit). Checks once
// that the game's device calls the hooked CreateVertexShader.
void OnFrameStart(ID3D11DeviceContext* ctx);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line when something changed, and at the end with the refusal reasons

// qvfr_proxy_smoke --mvr-vs: the hook on the test device (shaders from the test exe are patched too).
bool TestInstall(ID3D11Device* dev);
void TestSetEnabled(bool on);
struct TestState {
    unsigned long long made = 0, patched = 0, refused = 0, nvapiFailed = 0, skipped = 0, faults = 0;
    unsigned maxOutputReg = 0;
};
TestState TestGet();

}  // namespace qvfr::mvrvs
