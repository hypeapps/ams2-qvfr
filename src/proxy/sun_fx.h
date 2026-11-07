// Screen-space sun effects of the focus views taken from outer (docs/ams2-renderer.md, "Godraye i flara słońca").
//
// The game draws crepuscular rays (CGodRayScreenEffect, "rendertoscreen" pass) and lens flares (CSunLensFlare sprites)
// additively into the view's HDR target, with sizes and positions in screen units of the view. A focus view (a zoomed
// part of outer) then gets rays and flares that do not match outer, and the focus window shows as a box. Instead:
// - outer L and outer R: each draw of these effects runs a second time into this module's accumulation texture of the
//   eye (same blend, no depth: the game draws them with depth off), so the texture holds what the effects added;
// - focus L and focus R: the game's draws of these effects are skipped, and one full-screen pass adds the eye's
//   accumulation into the focus HDR target, read at the same direction (the focus window's part of outer UV).
// The pass runs at the first effect draw of the focus group (the HDR target is bound) or, when the focus group draws
// none of them, at the first draw of the pass that follows the effects in outer (learned per frame; it must read the
// HDR target, e.g. the MSAA resolve).
// The effect shaders are recognised by the DXBC checksum of the bytecode the game creates them from (hooks on the
// device's CreatePixelShader and the context's PSSetShader, Draw and DrawIndexed, installed through a temporary WARP
// device at VR_Init). Unknown checksums (another game build): nothing changes.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

namespace qvfr::sunfx {

// VR_Init in quad mode, once per process: the D3D11 hooks. Guards itself; false: the module stays off.
bool Install();
// Render thread, every frame start: `ctx` = the game's immediate context (null before the first Submit), `enabled` =
// setting and hotkey. Drops this module's D3D11 objects when the device changes.
void OnFrameStart(ID3D11DeviceContext* ctx, bool enabled);
// Eye groups (render thread): group 0 outer L, 1 outer R, 2 focus L, 3 focus R. A focus group gives the window it
// renders as a part of outer's UV (origin and size; the whole view = 0, 0, 1, 1).
void BeginGroup(int group, float originX, float originY, float scaleX, float scaleY);
void EndGroup();
// From quad's OMSetRenderTargets detour, after the real call: the bound render target 0 (only compared).
void OnSetTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv);

// DLAA (dlaa.h) in quad mode: one hook per D3D11 function, so DLAA's check of each DrawIndexed runs from this module's
// hook, before anything else. replace(c, count) true = DLSS replaced the draw (not forwarded); inside() true = a draw
// of DLSS itself (forwarded untouched); after(c) runs after each forwarded DrawIndexed. False: this module's draw hooks
// are not in place.
using DrawReplaceFn = bool (*)(ID3D11DeviceContext* c, UINT count);
using InsideFn = bool (*)();
using AfterFn = void (*)(ID3D11DeviceContext* c);
bool SetDrawReplace(DrawReplaceFn replace, InsideFn inside, AfterFn after);

// MVR pixel shader patch (mvr_ps.h): one hook per D3D11 function, so the game's pixel shaders made through this module's
// CreatePixelShader hook go through create(..., original, caller) instead of the original function (`caller`: the
// return address of the game's call). False: the hook is not in place.
using CreatePsFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                               ID3D11PixelShader**);
using CreatePsFilterFn = HRESULT (*)(ID3D11Device* dev, const void* code, SIZE_T len, ID3D11ClassLinkage* link,
                                     ID3D11PixelShader** out, CreatePsFn original, uintptr_t caller);
bool SetCreatePsFilter(CreatePsFilterFn create);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line, with the session totals at the end
void Shutdown();             // VR_Shutdown: D3D11 objects released (hooks stay)

// qvfr_proxy_smoke --sun-fx: hooks on the test device and pixel shaders registered as effect shaders.
enum Kind { KindNone = 0, KindRays = 1, KindFlare = 2 };
bool TestInstall(ID3D11Device* dev, ID3D11DeviceContext* ctx);
void TestRegister(ID3D11PixelShader* ps, Kind kind);
struct TestState {
    unsigned long long replays = 0, skipped = 0, composites = 0, anchorComposites = 0, missed = 0, faults = 0;
};
TestState TestGet();

}  // namespace qvfr::sunfx
