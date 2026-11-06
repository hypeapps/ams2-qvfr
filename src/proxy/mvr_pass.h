// NVIDIA Multi-View Rendering in the game, gate B7 (docs/mvr.md): outer L's G-buffer pass through MVR, as a test.
//
// With [debug] mvr_gbuffer_test=1 (needs mvr_vs_patch=1), in every shot frame the engine's run of outer L's
// CPhase1Render goes through Multi-View Rendering: every render target and depth target the pass binds or clears goes
// to a two-slice copy (Texture2DArray, views on both slices), MVR is on with 2 views, and before every draw with a
// patched VS its constant buffer slot 3 holds view 1's matrix M and clip planes (the game's binding goes back after
// the draw). View 1 is focus L: its projection (the focus window of the left eye, as the focus group renders it) in
// the top-left corner of slice 1 at focus L's render size, cut by the window's clip planes. Draws with a VS the patch
// did not make go with one view. After the pass slice 0 is copied into the game's targets and the game's views are
// bound again, so the frame goes on as without MVR. The task runs once: it replays software command buffers, and a
// replay releases a reference of its buffer (running it twice crashed the game in the first B7 build).
//
// The shot gets per target: the game's target after the copy back, slices 0 and 1 of the copy, and focus L's own after
// its CPhase1Render. Pairs: slice 0 vs the game's (the copy back) and focus L vs the corner of slice 1.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

#include "shots.h"

namespace qvfr::mvrpass {

// VR_Init in quad mode, once per process, with the VS patch installed and the executor patched. False: off.
bool Install();
// First Submit: hooks on the instanced and indirect draws of the immediate context (the plain draws come from sun_fx).
void InstallD3DHooks(ID3D11DeviceContext* imm);
bool On();  // installed and not stopped (cheap)

// Render thread, every frame start before the shot is prepared: ends the test of the previous frame (its log line,
// the copies released).
void OnFrameStart();
// A shot starts in this frame: the test's image names go to names[base..], its pairs to pairs[0..]. Returns the
// number of images (0: no test in this shot, the reason is logged); *pairCount gets the number of pairs.
int PrepareShot(const char** names, int base, int max, shots::Pair* pairs, int maxPairs, int* pairCount);
void CancelShot();  // the shot did not start: the test is dropped without a log line
// Outer L's CBeginEye: the focus window of the left eye in this frame as a map from outer L's clip space,
// focus = (sx x + ox w, sy y + oy w, z, w). valid = false: no focus projection in this frame (no test).
void SetWindow(bool valid, double sx, double ox, double sy, double oy);
// Focus L's CBeginEye: whether the group renders with the focus projection (else the comparison is meaningless).
void FocusProjected(bool projected);

// The executor thunk (quad's RunTask) around the engine's Execute of every task. group: the task's eye group
// (0 outer L, 1 outer R, 2 focus L, 3 focus R, -1 none).
void BeforeTask(uint8_t* task, int group);
void AfterTask(uint8_t* task);

// D3D11 detours of the immediate context, before the real call (after the clones' mapping): the targets to bind or
// clear instead during the test run. MapTargets: false = bind as they are; rtvOut has room for n entries.
bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut);
ID3D11RenderTargetView* MapClear(ID3D11DeviceContext* c, ID3D11RenderTargetView* v);
ID3D11DepthStencilView* MapClear(ID3D11DeviceContext* c, ID3D11DepthStencilView* v);
// After the real OMSetRenderTargets with what was bound: the targets of the observed passes.
void OnSetTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv);
// Draw detours, before the real draw: true = call AfterDraw after it.
bool BeforeDraw(ID3D11DeviceContext* c);
void AfterDraw(ID3D11DeviceContext* c);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // at the end of the session: tests run
void Shutdown();             // VR_Shutdown: a test left over is dropped

// qvfr_proxy_smoke --mvr-pass: the test run on a fake task without the engine (no shot images).
void TestSetup(ID3D11DeviceContext* ctx, const void* taskVtable);
// Arms the test for the next BeforeTask of a task with that vtable in group 0: outer size w x h, focus size fw x fh.
void TestArm(uint32_t w, uint32_t h, uint32_t fw, uint32_t fh, double sx, double ox, double sy, double oy);
ID3D11Texture2D* TestArray(int k);  // the copy of target k (not referenced), null when none
struct TestState {
    unsigned targets = 0, binds = 0, clears = 0, mapFailures = 0, patchedDraws = 0, plainDraws = 0, cbSwaps = 0;
    unsigned copiedBack = 0, viewsFailures = 0;
    bool ran = false, faulted = false;
};
TestState TestGet();
void TestEnd();  // the report line and the copies released (as at the next frame start)

}  // namespace qvfr::mvrpass
