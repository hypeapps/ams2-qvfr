// NVIDIA Multi-View Rendering of the G-buffer in the game, gate B8 (docs/mvr.md).
//
// With [debug] mvr_gbuffer=1 (needs mvr_vs_patch=1 and the executor patch), each frame the engine's run of the G-buffer
// pass (CPhase1Render) of outer L and of outer R also draws that eye's focus:
// - every render target and depth target the pass binds or clears goes to a two-slice copy (Texture2DArray, views on
//   both slices); MVR is on with 2 views and VS constant buffer slot 3 holds view 1's matrix M and clip planes;
// - view 1 is the eye's focus: its projection (the focus window, with focus's own TAA jitter instead of outer's) in
//   the top-left corner of slice 1 at the focus clones' size, cut by the window's clip planes;
// - a VS the patch did not make draws with one view (VSSetShader hook); a game call that clears VS slot 3 gets the
//   module's buffer back (VSSetConstantBuffers hooks). No per-draw hook.
// The MVR part ends at the pass's first bind without an outer-size target (the sun masks at 1/4 size, which read the
// linear depth) or with the task: MVR off, slice 0 copied into the game's targets, the corner of slice 1 into the
// eye's stash in the focus clones' formats (colors by copy, the depth by a short pass with SV_Depth). The rest of the
// task draws the masks into the game's targets without MVR; they are stashed too.
// The focus group of that eye then skips its CPhase1BuildCommandThread and CPhase1Render (Finish only, as in outer
// only mode: the pair builds and replays the group's own command buffers, so the replay count stays balanced) and gets
// the stash in its clones instead: the G-buffer as copied, the masks as outer's read in the focus window (the sun
// effects of the focus views come from outer anyway, sun_fx.h). Each task runs at most once per frame (a replay
// releases a reference of its buffer; running one twice crashed the game in the first B7 build).
// Without clones ready, the focus projection or the eye's stash of this frame, the focus group runs its own pass.
// In a shot frame the focus groups run their own pass and the shot compares focus L's with the stash.
#pragma once

#include <d3d11.h>

#include <cstdint>
#include <string>

#include "shots.h"

namespace qvfr::mvrpass {

using JitterFn = bool (*)(float out[2]);
// The engine's TAA projection offset of the current frame (dlaa::FrameJitter); without it view 1 keeps outer's jitter.
void SetJitterSource(JitterFn fn);
// VR_Init in quad mode, once per process, with the VS patch installed and the executor patched. False: off.
bool Install();
// First Submit: the module's D3D11 hooks on the immediate context (VSSetShader, VSSetConstantBuffers(1), the
// instanced and indirect draws; the plain draws come from sun_fx).
void InstallD3DHooks(ID3D11DeviceContext* imm);
bool On();  // installed and not stopped (cheap)

// Render thread, every frame start before the shot is prepared: `enabled` = the setting and the hotkey. Reports the
// shot comparison of the previous frame.
void OnFrameStart(bool enabled);
// A shot starts in this frame: the comparison's image names go to names[base..], its pairs to pairs[0..]. Returns the
// number of images (0: none in this shot, the reason is logged); *pairCount gets the number of pairs.
int PrepareShot(const char** names, int base, int max, shots::Pair* pairs, int maxPairs, int* pairCount);
void CancelShot();  // the shot did not start: the comparison is dropped without a log line
// Outer L's (eye 0) and outer R's (eye 1) CBeginEye: the eye's focus window in this frame as a map from outer's clip
// space, focus = (sx x + ox w, sy y + oy w, z, w). valid = false: no focus projection in this frame. w x h: outer's
// target size (the game's render size; 0: not checked).
void SetWindow(int eye, bool valid, double sx, double ox, double sy, double oy, uint32_t w, uint32_t h);
// Focus L's (eye 0) and focus R's (eye 1) CBeginEye: whether the group renders the focus window.
void FocusGroup(int eye, bool projected);
// Skip (default): the focus groups skip their G-buffer pass and get the stash, its depth made conservative
// ([debug] mvr_depth_dilate, mvr_depth_slope, mvr_depth_bias). Diagnosis modes (hotkey cycles): SkipRaw the same with
// the depth as MVR drew it (it flickers); the others run focus's own pass and then: NoDepth the stash without its depth,
// OuterOnly nothing, Overwrite the whole stash (tests).
enum class Mode { Skip, SkipRaw, NoDepth, OuterOnly, Overwrite };
void SetMode(Mode m);
Mode GetMode();
const char* ModeName(Mode m);

// The executor thunk (quad's RunTask) before the engine's Execute of every task. group: the task's eye group
// (0 outer L, 1 outer R, 2 focus L, 3 focus R, -1 none). True: do not execute it, call its Finish instead.
bool BeforeTask(uint8_t* task, int group);
void AfterTask(uint8_t* task);  // after the engine's Execute (not after a skipped one)

// MVR forward, step 1 (docs/mvr.md B9a), [debug] mvr_early: focus L's SSAO and lights (mode 2: SSAO only) run in
// outer L's group, right
// before its CPhase3Render (the forward pass that a later step draws with MVR), and the focus group skips them. Only
// focus L: the focus clones are shared, and outer R's group (which would run focus R's) comes before focus L's group.
// The run order of quad's RunTask:
//   EarlyWanted(eye) -> quad swaps focus's view in and puts focus's clones in -> EarlyBegin(eye, projected) -> the
//   engine's CBeginEye, CSSAORender, CPhase2Render of the focus group without their Finish -> clones out, view back,
//   outer's CBeginEye again without its Finish -> EarlyEnd(eye, ...).
enum class Kind { Other, Phase1, Build, Ssao, Phase2, Phase3 };
Kind KindOf(uint8_t* task);  // by vtable; Other when not installed
void SetEarly(int mode);     // the setting and its hotkey: 0 off, 1 SSAO and lights, 2 SSAO only, 3-5 diagnosis
constexpr int kEarlyModes = 6;
int GetEarly();
const char* EarlyName(int mode);
// True when this frame's eye should get its focus's SSAO and lights now (setting, B8 on and in skip mode, outer's
// stash from this frame, no shot). Counts the reason when not.
bool EarlyWanted(int eye);
// After the swap, with focus's clones in: the focus group's choice is made now (skip its G-buffer pass, SSAO and
// lights) and the stash goes into the clones. False: not now (the reason is counted); the focus group decides at its
// own time as in B8.
bool EarlyBegin(int eye, bool projected);
void EarlyRefused(int eye, const char* why);  // quad could not swap (the reason is counted)
void EarlyEnd(int eye, double ms, bool ok);   // ok = false: an engine fault in the early tasks
void EarlyWaited(double ms);                  // time quad waited for the engine's recording workers before the swap
// Task map of a shot frame or of a map frame (render thread): every task of the eye groups with its draws, binds,
// clears and the distinct targets it bound, one log line per task that touched the immediate context; `detail` adds
// the distinct PS shader resources and constant buffers of its draws (resource ids, clones marked).
void TaskMapBegin(const char* group, const char* name, bool detail);
void TaskMapEnd();

// D3D11 detours of the immediate context, before the real call (after the clones' mapping): the targets to bind or
// clear instead during the MVR part. MapTargets: false = bind as they are; rtvOut has room for n entries.
bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut);
ID3D11RenderTargetView* MapClear(ID3D11DeviceContext* c, ID3D11RenderTargetView* v);
ID3D11DepthStencilView* MapClear(ID3D11DeviceContext* c, ID3D11DepthStencilView* v);
// Draw detours, before the real draw: true = call AfterDraw after it (draw counts in a shot frame only).
bool BeforeDraw(ID3D11DeviceContext* c);
void AfterDraw(ID3D11DeviceContext* c);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line per interval, the session totals at the end
void Shutdown();             // VR_Shutdown: textures released

// qvfr_proxy_smoke --mvr-pass: fake tasks and fake clones on a test device (no shot images).
using CloneFn = ID3D11Texture2D* (*)(ID3D11Resource* original);
void TestSetup(ID3D11DeviceContext* ctx, const void* phase1Vtable, const void* buildVtable, CloneFn clone,
               JitterFn jitter);
struct TestState {
    unsigned passes = 0, focusSkipped = 0, fills = 0, binds = 0, clears = 0, mapFailures = 0, patchedSets = 0;
    unsigned plainSets = 0, cbRebinds = 0, copiedBack = 0, stashed = 0, masks = 0, viewsFailures = 0, failures = 0;
    bool stashValid = false, jitterKnown = false, faulted = false, lateStart = false;
    float rows[2][4] = {};  // M's x and y rows of the last pass
};
TestState TestGet();
ID3D11Texture2D* TestArray(int k);
// Overrides mvr_depth_bias, mvr_depth_slope and mvr_depth_dilate (steps < 0: the settings).
void TestDepthBias(int steps, float slope, bool dilate);  // the two-slice copy of target k (not referenced), null when none
void TestEnd();                     // the next frame start (counters of the frame reset)

}  // namespace qvfr::mvrpass
