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
// S1 (docs/dlaa-plan.md, sr.h): stereo with the outer set at the render size. Only the scene phase of an eye group
// (until the TAA resolve's draw) renders into the clones. In the learning groups MarkTaa at the TAA draw ends the scene
// phase: a texture gets a clone when it is written before it and is not the TAA draw's target; one used after it too is
// carried (its clone's contents go into the original at the TAA draw).
void SetStereoSr(bool on);                    // before the first frame
void MarkTaa(ID3D11DeviceContext* imm);       // learning: the TAA draw comes now (its target bound)
// While the outer clones are in: the carried textures' clones and originals (not referenced; valid until frame start).
int CarryList(ID3D11Texture2D** clone, ID3D11Resource** orig, int max);
// At the TAA draw while the clones are in: they go out as at CEndEye (fields, size constants, viewport, the bound
// state); until the group ends the bind fix sends views of clones to their originals. False: no clones were in.
bool SceneEnd(ID3D11DeviceContext* imm);
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
// Focus R's own clones (MVR gate B9d). Focus L and R share one set of clones; a focus clone that outer's groups write
// for an eye before the focus groups run (the early SSAO and lights, the live forward pass) gets a second texture, the
// twin, for focus R. While eye 1 is chosen, focus's swap, CloneOf(…, Focus) and the bind fix give the twin in its place
// (clones without a twin stay shared). Original() of a twin is the engine's texture.
// Render thread, at the start of each eye group: the focus eye of the work that follows (0 L, 1 R).
void SetFocusEye(int eye);
// Render thread: `original`'s focus clone gets a twin at the next frame start (no-op without a focus clone).
void WantTwin(ID3D11Resource* original);
// Render thread: `original`'s focus clone has its twin now.
bool HasTwin(ID3D11Resource* original);
// MVR gate B17 (docs/mvr.md): two-slice arrays in place of the engine's own textures. mvr_pass asks for the targets of
// outer's G-buffer pass (B12: and the multisampled ones of its forward pass, the arrays then multisampled too); at the
// next frame start each listed texture whose object still holds it gets a Texture2DArray
// of 2 slices with its description (a depth format also with a shader view bind), slice 0 starting with its contents.
// The array and views like the engine's on it (a TEXTURE2D view of an array is slice 0) go into the texture object's
// fields and the render targets' view arrays, so the engine draws and reads slice 0 itself and MVR's view 1 needs no
// copy back. The engine's own texture and views stay alive: a hooked call that still reaches them (a view kept outside
// the objects) goes to the array (a view made on it when needed), and CopyResource between an array and a single
// texture (or two arrays) copies slice 0 only. Before the engine destroys an object, at a new game size, at VR_Shutdown
// and on a fault the engine's own values go back. Made arrays drop the clones once: they are made again from the new
// views (single textures as before). Render thread.
// B12 step 5 (docs/mvr.md, [debug] mvr_ms_slices): `slices` 3 for the forward pass's multisampled targets. The focus
// clone of a three-slice array is then the array itself: focus L's swapped fields and translated views see slice 1,
// focus R's (its twin) slice 2, in the focus clones' size (the engine's size fields and the viewport check say focus's
// size; the corner at the top left). No texture of its own, nothing copied when the twin is made.
void WantArray(ID3D11Resource* original, UINT slices = 2);
// Any thread, lock-free: `res` is one of these arrays.
bool IsArray(ID3D11Resource* res);
// Render thread: the slice of `original`'s array that focus's work of `eye` uses as its clone (1 focus L, 2 focus R
// with its twin, 1 for focus R before its twin is made), 0 when the focus clone is a texture of its own (or none).
// w, h: focus's size of the slice clone (when not null and the result is not 0).
int FocusSlice(ID3D11Resource* original, int eye, uint32_t* w = nullptr, uint32_t* h = nullptr);
// Any thread: changes whenever the clones are dropped (views kept on old clones would keep them alive).
uint32_t Generation();
// The engine device's size constants (device from the factory hook): the render target size of command contexts
// 0..3 (+0x9b18, stride 0xe60) and the postprocess object's scene size (+0x100). False: no device yet or not
// readable. S1 probe (mvr_pass task map).
bool EngineSizes(float rt[4][2], int post[2]);
// Render thread: while one lives, this thread's D3D11 calls pass the clones' hooks as they are (no bind fix, not
// observed). The MVR forward test (gate B9c) copies focus's clones inside outer's group, where the bind fix would put
// the originals in their place.
class RawScope {
public:
    RawScope();
    ~RawScope();
    RawScope(const RawScope&) = delete;
    RawScope& operator=(const RawScope&) = delete;

private:
    bool prev_;
};
// Render thread: the clones are made, wanted in this frame and their list unchanged (a group of a set with clones
// would get them now). MVR forward step 1 swaps focus's in inside outer's group only then (not while learning).
bool IsReady();
// For a shot: up to `max` clones of `set` (not referenced; valid until the next frame start) with a label each
// ("<set>_NN_<w>x<h>_f<format>_<role>", NN as in the list order of the log). Returns the count.
int ListForShot(Set set, ID3D11Resource** out, char (*labels)[64], int max);

std::string Describe();      // one part of the summary line
void LogStats(bool totals);  // a stats line, with the session totals at the end
// O1 ([debug] clones_map_cache, live): binds whose views all mapped to themselves since the latest change of the maps
// skip the lock (a cache on the render thread).
void SetMapCache(bool on);
bool GetMapCache();
// O1: PSSetShaderResources calls on the immediate context that reach slot kPsWatchSlot or above (any caller, the
// proxy's own too), counted: mvr_pass binds its t21..t26 (mvrps::kTexBase) again only when they may have changed.
constexpr UINT kPsWatchSlot = 21;
uint32_t PsHighSlotsGen();
// O1 step 3: the view at PS slot `slot` of context `c` without PSGetShaderResources when the PSSetShaderResources hook
// saw that slot's latest bind on the immediate context since the latest NewPsViewEpoch (else read back once and kept for
// the epoch). Not referenced: valid while bound. Render thread. A new epoch at each MVR pass's begin covers any state
// change the hook does not see (SwapDeviceContextState of DLSS, ClearState), none of which runs inside a pass.
void NewPsViewEpoch();
ID3D11ShaderResourceView* PsView(ID3D11DeviceContext* c, UINT slot);
uint64_t PsViewReads();  // PsView calls that read the slot back, so far
// qvfr_proxy_smoke --clones: the map cache's binds without and with the lock so far; true while a group watches
// (learning or proof: no cache).
bool TestMapCache(uint64_t& hits, uint64_t& locked);
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
    int twins = 0, twinViews = 0;
    int arrays = 0;  // B17
    uint64_t arrayFixes = 0, arrayLazy = 0, arrayCopies = 0;
};
TestState TestGet();

}  // namespace qvfr::clones
