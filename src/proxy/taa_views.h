// Separate TAA state of the focus views (docs/dlaa-plan.md, D2).
//
// The engine's TAA (CTAAControl) keeps its per-view state in arrays of 3 view slots: history (two targets, flip-flop
// by frame), velocity, previous depth, locks (two targets), each a render target object with its texture, and the
// previous camera matrices. Outer R and both focus groups run as view slot 2, so without this module they share one
// history: the focus views reproject outer R's image and smear.
//
// When the engine creates a slot 2 target (CTAAControl ctor, through its target helper), a hook calls the same helper
// twice more for this module's focus L and focus R copies: real engine objects, so the clones module lists them like
// the engine's own. While a focus group runs, the slot 2 entries of the arrays and the slot 2 previous matrices hold
// that focus view's own; after the group outer R's go back. The engine's TAA release frees the copies with the same
// engine function it uses for its own targets.
//
// The TAA resolve and locks shaders read their pixels at uv * viewportScale.zw, and the engine keeps that constant at
// one place for all views. With clones the views differ in size, so a view got another's size (outer showed a zoomed
// image). Before each of these draws the module binds its own cb0 with the size of the bound render target.
// In a focus group the object velocity draws get the focus view's previous projection and view-projection (the engine
// gives them this frame's focus window, so moving objects smeared while the window followed the gaze).
#pragma once

#include <d3d11.h>

#include <string>

namespace qvfr::taaviews {

// VR_Init in quad mode, after the builder patch, once per process: hooks on the TAA target helper and release.
// Guards itself; false: the module stays off (the focus views share outer R's TAA state).
bool Install();
// Render thread, every frame start: `enabled` = setting and hotkey (a swap left over goes back). `dump`: this frame is
// a shot frame; the full-screen draws of outer R and focus L are logged with their bound views (diagnostics).
void OnFrameStart(bool enabled, bool dump);
// Eye groups on the render thread: 0 outer L, 1 outer R, 2 focus L, 3 focus R (only the focus groups swap).
// BeginGroup before the engine's CBeginEye Execute, EndGroup after the engine's CEndEye Execute.
void BeginGroup(int group);
void EndGroup();
// From sun_fx's Draw and DrawIndexed detours, before the real call: the shot-frame dump and the TAA viewportScale fix
// (the engine keeps the TAA resolve's and locks' viewport size in one place for all views; with clones the views
// differ in size) and the object velocity fix of the focus groups. True: a constant buffer binding was replaced (PS slot
// 0 or VS slot 12); call AfterDraw after the real draw.
bool OnDraw(ID3D11DeviceContext* c, bool indexed, UINT count);
void AfterDraw(ID3D11DeviceContext* c);
// From quad's OMSetRenderTargets detour, after the real call: whether a velocity buffer is bound (object motion vector
// fix in the focus groups).
void OnSetTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv);
// qvfr_proxy_smoke --taa-fix: the fix without the engine hooks, on the test's context.
void TestEnableFix(ID3D11DeviceContext* c, bool on);
uint64_t TestFixedDraws();  // resolve and locks draws fixed so far
// As in a focus group with its previous matrices (null: none), and the object velocity draws fixed so far.
void TestSetPrevFocus(const float proj[16], const float viewProj[16]);
uint64_t TestObjectDraws();

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line, with the session totals at the end
void Shutdown();             // VR_Shutdown: a swap left over goes back (the copies stay with the engine's TAA)

}  // namespace qvfr::taaviews
