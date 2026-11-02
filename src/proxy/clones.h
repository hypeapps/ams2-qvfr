// Clones of the engine's eye-size targets for the focus groups (M7.3, docs/ams2-renderer.md).
//
// Hooks on the engine's texture and render target factory list every texture and render target in the game's render
// size or its 1/2, 1/4, 1/8 (with names); the destructor hooks drop them. In the first focus groups of a set (learning)
// the D3D11 hooks note which of them the focus groups write: only those get a clone (M7.3: same size). A texture the
// focus groups only read comes from elsewhere (outer, the passes once per frame) and stays shared.
//
// While a focus group runs, the engine objects hold the clone's D3D11 texture and views instead of their own (texture
// +0x58 and its views, the render targets' view arrays), and the D3D11 state bound at that moment is translated the same
// way, so a bind the engine skips as redundant still reaches the clone. After the group everything goes back. In the
// first focus groups with clones (proof) the hooks count accesses to the originals of cloned targets (none expected)
// and to the clones.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::clones {

// VR_Init on the supported exe (quad mode): factory and destructor hooks. False: no clones in this process.
bool InstallEngineHooks();
// First Submit: D3D11 hooks on the immediate context (they only count in learning and proof groups).
void InstallD3DHooks(ID3D11DeviceContext* imm);
// The game's render size of this VR session (0 = none): targets in this size and its fractions are listed.
void SetEyeSize(uint32_t w, uint32_t h);

// Render thread. Each call guards itself: a fault turns clones off for the session and puts the originals back.
// Frame start: a swap left over is put back, new objects are sorted, a changed set starts learning again, the clones are
// made after learning. `enabled`: clones wanted now (setting, hotkey, focus window on).
void OnFrameStart(ID3D11DeviceContext* imm, bool enabled);
void BeginFocus(ID3D11DeviceContext* imm);  // focus group CBeginEye, before the engine's Execute
void EndFocus(ID3D11DeviceContext* imm);    // focus group CEndEye, after the engine's Execute (no-op without BeginFocus)
// From quad's OMSetRenderTargets hook (render target binds count as writes).
void OnOMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                          ID3D11DepthStencilView* dsv);
// The engine's texture behind a clone, or `res` itself (the focus groups' output may be the clone of the Submit texture).
void* Original(void* res);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line, with the session totals at the end
void Shutdown();             // VR_Shutdown: originals back, clones released, lists cleared (hooks stay)

// qvfr_proxy_smoke --clones: fake engine objects without the engine hooks.
void TestRegister(bool renderTarget, void* obj);
void TestDestroy(bool renderTarget, void* obj);
struct TestState {
    int phase = 0;  // 0 waiting, 1 learning, 2 ready, 3 failed
    int textures = 0, renderTargets = 0, clones = 0, entries = 0;
    bool swapped = false;
    uint64_t cloneUses = 0, leaks = 0, translated = 0;
};
TestState TestGet();

}  // namespace qvfr::clones
