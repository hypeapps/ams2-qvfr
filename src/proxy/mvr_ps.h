// Pixel shaders of the game made ready for an extra Multi-View Rendering view (docs/mvr.md, gate B9b).
//
// A hook on the device's CreatePixelShader patches every pixel shader the game creates that reads screen inputs or
// screen textures (qvfr_dxbc PatchPsForMvr: in a non-zero view, screen inputs get view 1's transform from PS constant
// buffer slot 3 and the screen textures globalTexture0..4 come from slots 21..25). View 0 draws exactly what the
// original shader draws, so the game's image does not change while MVR is off; the slots need no binding until then.
// A refused patch gives the game the original shader and a log entry with the reason. Only devices with Multi-View
// support (mvrvs) get patched shaders.
//
// The patched shaders take the view number from SV_RenderTargetArrayIndex, but only while their constant buffer holds
// the magic value (dxbc::kMvrPsMagic in row 2.x), which only the proxy's MVR passes bind. The game has geometry
// shaders that write the index (layered targets, B9b in the game): behind them the patched shaders draw as the
// original. The module still counts such shaders (geometry shaders here, vertex shaders in mvr_vs) and logs the first.
//
// [debug] mvr_ps_patch=1 turns it on (default off; needs mvr_vs_patch=1; a change needs a game restart). It runs in
// the sun effects module's CreatePixelShader hook (sunfx::SetCreatePsFilter; one hook per D3D11 function) and hooks
// CreateGeometryShader (both variants) itself, through a temporary WARP device, at VR_Init in quad mode after mvr_vs.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::mvrps {

constexpr unsigned kCbSlot = 3;     // PS constant buffer of the extra view (free in the scene's PS, docs/mvr.md B2)
constexpr unsigned kTexBase = 21;   // the extra view's screen textures (the scene's PS use t0..t20)
// Screen textures: globalTexture0..4 (0..4) and the live track map (5: rubber, dirt and skid marks the eye group's
// CLiveTrackRender draws at the eye's size; the track's PS sample it at their screen UV, docs/mvr.md B9d run 5).
constexpr unsigned kTextures = 6;
constexpr unsigned kLiveTrack = 5;
constexpr const char* kLiveTrackName = "GLOBAL_liveTrackTextureMap_tex";

// VR_Init in quad mode with the setting on (after sunfx::Install and mvrvs::Install), once per process. False: the
// module stays off.
bool Install();
// Render thread, every frame start: checks once that the game's device calls the hooked CreatePixelShader.
void OnFrameStart(ID3D11DeviceContext* ctx);

enum class Kind {
    Unknown,    // not made through the hook (or null)
    Patched,    // the patched shader
    NotNeeded,  // the original: it reads no screen input and no screen texture
    Original,   // the original although it needs the patch (refused); wrong in an extra view
};
// Any thread.
Kind KindOf(ID3D11PixelShader* ps);
// A patched shader's game slots of globalTexture0..4 (kNoSlot: not read): the MVR pass binds the extra view's copy of
// the texture in slot k at kTexBase + k (gate B9c).
constexpr uint8_t kNoSlot = 0xFF;
struct Info {
    Kind kind = Kind::Unknown;
    uint8_t slot[kTextures] = {kNoSlot, kNoSlot, kNoSlot, kNoSlot, kNoSlot, kNoSlot};
};
Info InfoOf(ID3D11PixelShader* ps);
bool On();  // installed and not stopped

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line when something changed, and at the end with the refusal reasons

// qvfr_proxy_smoke --mvr-ps: after sunfx::TestInstall and mvrvs::TestInstall; the test exe's own shaders get patched.
bool TestInstall();
void TestSetEnabled(bool on);
struct TestState {
    unsigned long long made = 0, patched = 0, notNeeded = 0, refused = 0, failed = 0, gsMade = 0, gsLayered = 0,
                       faults = 0;
};
TestState TestGet();

}  // namespace qvfr::mvrps
