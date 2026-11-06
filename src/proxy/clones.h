// Clones of the engine's eye-size targets: smaller targets for outer or focus (M7.3, M7.4; docs/ams2-renderer.md).
//
// The game renders at the larger of the outer and focus sizes per axis. A set of views (outer or focus) whose own size
// differs from it renders into clones in its own size. Hooks on the engine's texture and render target factory list
// every texture and render target in the game's render size or its 1/2, 1/4, 1/8; the destructor hooks drop them. In
// the first eye groups of a list (learning) the D3D11 hooks note which of them outer and focus write: a set gets clones
// of the targets it writes. Outer keeps a target focus reads but does not write (focus needs outer's data).
//
// While a group of that set runs, the engine objects hold the clone's D3D11 texture, views and size instead of their
// own (texture +0x58, its views, size +0x68; render target view arrays and size +0x24), the device's render target size
// of each command context (+0x9b18) and the postprocess object's scene size (device+0x17b0, +0x100) say the clone's
// size, and the D3D11 state bound at that moment is translated the same way. A viewport set for an original while a
// clone is bound is counted (and scaled with [debug] clones_viewport_fix). Engine passes that keep their own D3D11
// views of a target are caught in D3D11: in a group with clones a bind, clear or copy of a cloned original goes to its
// clone (a view made on the clone when needed), in the other eye groups a clone goes back to its original ([debug]
// clones_bind_fix). After the group everything goes back.
// In the first groups with clones (proof) the hooks count accesses to the originals of cloned targets that were not
// translated (none expected) and to the clones.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::clones {

enum Set { Outer = 0, Focus = 1 };

// VR_Init on the supported exe (quad mode): factory and destructor hooks. False: no clones in this process.
bool InstallEngineHooks();
// First Submit: D3D11 hooks on the immediate context (they only count in learning and proof groups, and track the
// bound target and the viewport while clones are in).
void InstallD3DHooks(ID3D11DeviceContext* imm);
// The sizes of this VR session (0 = none): the game's render size (targets in it and its fractions are listed) and
// what outer and focus want. A set whose size differs from the game's gets clones.
void SetSizes(uint32_t gameW, uint32_t gameH, uint32_t outerW, uint32_t outerH, uint32_t focusW, uint32_t focusH);

// Render thread. Each call guards itself: a fault turns clones off for the session and puts the originals back.
// Frame start: a swap left over is put back, new objects are sorted, a changed list starts learning again, the clones
// are made after learning. `enabled`: clones wanted now (setting, hotkey, focus window on).
void OnFrameStart(ID3D11DeviceContext* imm, bool enabled);
void BeginGroup(ID3D11DeviceContext* imm, Set set);  // eye group CBeginEye, before the engine's Execute
void EndGroup(ID3D11DeviceContext* imm);             // eye group CEndEye, after the engine's Execute (no-op without Begin)
// From quad's OMSetRenderTargets hook, before the real call: the targets to bind instead of the engine's (bind fix;
// `rtvOut` holds 8 entries). False: bind the engine's as they are.
bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut, void* caller);
// From quad's OMSetRenderTargets hook, after the real call with what was bound (render target binds count as writes).
// `caller`: the engine's return address.
void OnOMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                          ID3D11DepthStencilView* dsv, void* caller);
// The engine's texture behind a clone, or `res` itself (a group's output may be the clone of a Submit texture).
void* Original(void* res);
// Render thread (MVR gate B8): the clone of the engine's texture `res` in `set` (not referenced; valid until the next
// frame start), null when it has none or the clones are not ready.
ID3D11Texture2D* CloneOf(ID3D11Resource* res, Set set);
// Render thread: a group of `set` runs now with its clones in the engine objects.
bool SwappedIn(Set set);
// For a shot: up to `max` clones of `set` (not referenced; valid until the next frame start) with a label each
// ("<set>_NN_<w>x<h>_f<format>_<role>", NN as in the list order of the log). Returns the count.
int ListForShot(Set set, ID3D11Resource** out, char (*labels)[64], int max);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line, with the session totals at the end
void Shutdown();             // VR_Shutdown: originals back, clones released, lists cleared (hooks stay)

// qvfr_proxy_smoke --clones: fake engine objects without the engine hooks.
void TestRegister(bool renderTarget, void* obj);
void TestDestroy(bool renderTarget, void* obj);
void TestSetDevice(void* device);  // fake engine device (size per context at +0x9b18, postprocess object at +0x17b0)
struct TestState {
    int phase = 0;  // 0 waiting, 1 learning, 2 ready, 3 failed, 4 not needed
    int textures = 0, renderTargets = 0, entries = 0;
    int clones[2] = {};
    bool swapped = false;
    uint64_t cloneUses = 0, leaks = 0, translated = 0, viewportsFixed = 0, viewportMismatches = 0;
    uint64_t bindsToClone = 0, bindsToOriginal = 0, lazyViews = 0;
};
TestState TestGet();

}  // namespace qvfr::clones
