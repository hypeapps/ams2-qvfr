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
// With mvr_forward (step 3, below) focus R's run in outer R's group as well, with snapshots of the shared clones.
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
// MVR forward, step 2 (docs/mvr.md B9c), [debug] mvr_forward_test with mvr_early=1 and the PS patch (mvr_ps.h): a shot
// frame also runs outer L's forward pass (CPhase3Render, right after the early run) with MVR. Its targets go to the
// two-slice copies as in the G-buffer pass, each filled at its first bind or clear: slice 0 with the game's texture,
// the corner of slice 1 with focus's clone (focus L's G-buffer, SSAO and lights from the early run). Before each draw
// the module picks 2 views when the VS is patched and the PS is patched or needs no patch, else 1; a patched PS gets
// PS slot 3 (view 1's screen input map, the UV scale, the magic value) and the views of its screen textures on focus's
// clones in t21..t25. After the pass slice 0 goes back to the game's targets and the corners are kept; focus L's own
// CPhase3Render then runs as always and the shot compares its targets with them (images mvr_f<k>_focus and _ours, log
// lines "MVR forward test shot"). Nothing goes into focus's clones: the frame shows focus's own forward pass.
// A bind of targets of another size (the game's 1/2 size f9 and its depth) pauses MVR: one view into the game's
// targets until the next bind with an outer-size target; their clears go to the game's views too.
// MVR forward, step 3 (docs/mvr.md B9d), [debug] mvr_forward (live, hotkey mvr_modifier + mvr_forward_key; needs
// mvr_early=1, the PS patch and B8 in skip mode): every frame outer L's and outer R's CPhase3Render run with MVR right
// after their eye's early run (focus R's SSAO and lights then run early in outer R's group too). At the end of the pass
// view 1's corners go into focus's clones (colors by copy, the depth by mvr_forward_depth, below), the
// pass's clears of targets of other sizes (the 1/2 size f9) are repeated on their focus clones, and the focus group
// skips its CPhase3BuildCommandThread and CPhase3Render (Finish only). A pass with draws that miss view 1 (one view, or
// into targets of other sizes) leaves the clones alone and the focus group runs its own.
// The focus clones are shared by both eyes, so the module keeps two snapshots of the clones that the early runs and
// the forward passes write (noted as they are bound, cleared or filled): before outer R's early run it saves focus L's
// (snapshot L); at focus L's CBeginEye it saves focus R's and puts focus L's back; at focus R's CBeginEye it puts focus
// R's back.
void SetForward(bool on);
bool GetForward();
// The depth the live pass leaves in focus's clone ([debug] mvr_forward_depth): 0 as MVR drew it, 1 plus mvr_depth_bias
// float steps, 2 conservative like the G-buffer stash, 3 (default) as drawn, and while the focus group's velocity pass
// draws (an R16G16 target with the depth clone bound) a conservative copy (dilate, slope, bias) in its place. 1 and 2
// thin out the crowd in the stands (a later pass tests the crowd against this depth); with 0 focus's own geometry in
// the velocity pass misses the nearer MVR depth and the cockpit smears after DLAA while driving (docs/mvr.md B9d runs 3,
// 4).
constexpr int kForwardDepthModes = 4;
void SetForwardDepth(int mode);
int GetForwardDepth();
const char* ForwardDepthName(int mode);
// Quad: the vtable of the focus groups' CPhase3BuildCommandThread (found by class name; 0: none).
void SetPhase3BuildVtable(uintptr_t vt);
// A focus group's CBeginEye, after its clones went in (snapshots, above).
void FocusGroupIn(int eye);
// Task map of a shot frame or of a map frame (render thread): every task of the eye groups with its draws, binds,
// clears and the distinct targets it bound, one log line per task that touched the immediate context; `detail` adds
// the distinct PS shader resources and constant buffers of its draws (resource ids, clones marked).
void TaskMapBegin(const char* group, const char* name, bool detail);
void TaskMapEnd();

// D3D11 detours of the immediate context, before the real call (after the clones' mapping): the targets to bind or
// clear instead during the MVR part. MapTargets: false = bind as they are; rtvOut has room for n entries.
bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut);
// The clear's values: the live forward pass repeats its clears of other-size targets on their focus clones.
ID3D11RenderTargetView* MapClear(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT col[4]);
ID3D11DepthStencilView* MapClear(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT flags, FLOAT depth,
                                 UINT8 stencil);
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
// qvfr_proxy_smoke --mvr-forward: after OnFrameStart and SetWindow, arms the forward test of this frame without shot
// images, as if focus L's SSAO and lights had run early; `phase3Vtable` stands for CPhase3Render's.
void TestForward(const void* phase3Vtable);
struct TestForwardState {
    bool outerDone = false, focusDone = false, failed = false;
    std::string why;  // the pass's failure or why it did not run
    unsigned draws = 0, plainDraws = 0, psPatchedDraws = 0, psCbRebinds = 0, srvMisses = 0, copyIns = 0;
    unsigned binds = 0, mapFailures = 0, viewsFailures = 0, copiedBack = 0;
    unsigned pauses = 0, pausedDraws = 0, pausedClears = 0;  // binds of other sizes (MVR paused)
    int stashed = 0;
};
TestForwardState TestForwardGet();
ID3D11Texture2D* TestForwardStash(int k);  // the corner of slice 1 of forward target k (not referenced)
// qvfr_proxy_smoke --mvr-forward, live mode (B9d): after OnFrameStart and SetWindow, the live forward pass of eye L as
// if focus L's early run had gone (focus L decided to skip, its SSAO and lights done).
void TestForwardLive(const void* phase3Vtable);
// As if the eye's early run began now (eye 1: snapshot L first, as EarlyBegin does).
void TestEarlyMark(int eye);
struct TestLiveState {
    unsigned passes = 0, ok = 0, focusSkipped = 0, buildSkipped = 0, otherClears = 0, otherClearMisses = 0;
    unsigned snapEntries = 0, saves[2] = {}, restores[2] = {}, velBinds = 0;
    std::string why;  // the latest live pass's failure
};
TestLiveState TestLiveGet();

}  // namespace qvfr::mvrpass
