#include "quad.h"

#include <d3d11.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "clones.h"
#include "common.h"
#include "compose.h"
#include "dlaa.h"
#include "focus_cull.h"
#include "focus_geometry.h"
#include "gpu.h"
#include "hooks.h"
#include "mvr_pass.h"
#include "mvr_ps.h"
#include "mvr_vs.h"
#include "shots.h"
#include "sun_fx.h"
#include "taa_views.h"

namespace qvfr::quad {

namespace {

using hooks::Anchor;

// Engine layout from M3 and M4.1 (docs/ams2-renderer.md), AMS2AVX.exe build aa43bd68.
constexpr size_t kTaskCtx = 0x860;       // eye tasks: render context pointer
constexpr size_t kTaskView = 0x868;      // eye tasks: view number (int)
constexpr size_t kCtxSlots = 0x27bc8;    // per view {camera, camera, main camera}
constexpr size_t kSlotSize = 0x18;
constexpr size_t kCtxCamBlock = 0x190;   // per view camera data; the camera functor fills views 0..2 once per frame
constexpr size_t kCamBlockSize = 0x1a0;
constexpr size_t kCtxQuery = 0x70c;      // per view lighting environment index (CLightingEnvManager), stride 4
constexpr int kFocusView = 2;            // the view number the focus groups run as
constexpr size_t kAppType = 0x8;         // app singleton: type (5 or 7..9 with a VR device)
constexpr size_t kAppVr = 0xe0;          // app singleton: engine VR device (_BVROpen)
constexpr size_t kVrEyeData = 0xa0;      // VR device: per eye data (tangents, eye to head, projection), eye 0 first
constexpr size_t kVrEyeSize = 0x110;
constexpr size_t kEyeTangents = 0x0;     // eye data: 4 floats {b, -t, -l, r} computed from the projection (0xd6bf30)
constexpr size_t kEyeProj = 0x90;        // eye data: projection, row vectors, z forward (m20 = -OpenVR m02, m23 = 1)
constexpr uintptr_t kCamVtable = 0x2115d30;  // camera class vtable (RVA); all view cameras have it
constexpr size_t kCamView = 0x90;        // camera: view matrix, row vectors (+0x150 returns it times +0x110)
constexpr size_t kCamProj = 0x110;       // camera: explicit projection (set through vtable +0x210, read by +0x138/+0x150)
constexpr size_t kCamRotDirty = 0x400;   // camera: rebuild the rotation, the position (the view matrix is stale meanwhile)
constexpr size_t kCamPosDirty = 0x401;
constexpr size_t kCamProjDirty = 0x402;  // camera: 1 = rebuild the projection from FOV and aspect at the next update
constexpr size_t kMatSize = 0x40;
// Task system (M4.5, disassembly of 0xcd0cd0 and the executor 0xcd12a0): "add task" (rcx = group, rdx = task) puts
// the task into the group and writes the group to task+0x848, and the executor runs a task through vtable +0x48
// (Execute) at two call sites; state changes and dependents are handled by the executor around it. In the game
// CBeginEye and CEndEye of one eye group end up with different +0x848 after the builder (first M4.5 run), so the
// focus tasks are taken from the "add task" calls inside the builder instead.
constexpr size_t kTaskGroup = 0x848;  // only logged
constexpr int kExecuteSlot = 0x48 / 8;
constexpr uintptr_t kBeginEyeVtable = 0x204c940;  // CBeginEye vtable (RVA), Execute 0x9d7670 at +0x48
constexpr uintptr_t kEndEyeVtable = 0x204c838;    // CEndEye vtable (RVA), Execute 0x9d7900 at +0x48
constexpr size_t kMaxAdds = 1 << 16;              // "add task" calls recorded per builder call
constexpr int kIsGroupSlot = 0x28 / 8;            // vtable slot the executor asks first: true for task groups
constexpr uintptr_t kLeafIsGroup = 0x9bebf0;      // its leaf version (RVA): xor al, al; ret
// A leaf's Execute ends with `jmp [vtable+0x38]`: Finish (0xcde260) sets state 4 and releases the dependents and the
// group. A skipped task must still be finished, or the group's scheduler queues it again forever (third M4.5 run).
constexpr int kFinishSlot = 0x38 / 8;
constexpr uintptr_t kTaskFinish = 0xcde260;
// Task groups (vtable 0x2088310) answer true and their Execute (0xcd1180) runs the children's scheduler.
// Watchdog: about two seconds of frames in which the focus groups ran but no eye was composed.
constexpr int kWatchFrames = 144;

// Eye groups in build order. CBeginEye/CEndEye::Execute pick the tonemap target and the Submit texture by view - 1,
// so every group after outer L writes into the right eye's Submit texture and outer L's stays untouched.
constexpr int kGroups = 4;
enum Group { OuterL, OuterR, FocusL, FocusR };
const char* const kGroupNames[kGroups] = {"outer L", "outer R", "focus L", "focus R"};
// Shot images: the four views at their CEndEye, then the composed eyes at Submit.
constexpr int kComposed = kGroups;  // + eye
constexpr int kShotFixed = kGroups + 2;
constexpr int kShotImages = shots::kMaxImages;  // the rest: focus's clones at the end of focus L (M7.4 diagnostics)
char g_dumpNames[kShotImages - kShotFixed][64];
ID3D11Resource* g_dumpRes[kShotImages - kShotFixed] = {};  // not referenced: released no earlier than the next frame
const char* g_shotNames[kShotImages] = {"outer_l", "outer_r", "focus_l", "focus_r", "composed_l", "composed_r"};
int g_shotCount = kShotFixed;
// The view whose camera, frustum block and lighting environment each group renders with; the focus groups replace the
// projection (camera and VR device eye data) with the focus window of that eye and, with focus culling (M9.3), the draw
// queue with one culled for that window.
constexpr int kSourceView[kGroups] = {1, 2, 1, 2};
// Rectangles filled in by SetupLocked: the focus views against their window in the outer views and in the composed
// eyes (only inside the blend band, where the composition must be the focus pixel for pixel), and the outer views
// against the whole composed eye.
shots::Pair g_shotPairs[] = {
    {FocusL, OuterL, "focus L vs outer L"},
    {FocusR, OuterR, "focus R vs outer R"},
    {OuterL, OuterR, "outer L vs outer R (reference: parallax)"},
    {FocusL, kComposed + 0, "focus L vs composed L"},
    {FocusR, kComposed + 1, "focus R vs composed R"},
    {OuterL, kComposed + 0, "outer L vs composed L"},
    {OuterR, kComposed + 1, "outer R vs composed R"},
};
constexpr int kMvrPairs = 8;  // MVR gate B7 (mvr_pass.h): two per target
shots::Pair g_allPairs[std::size(g_shotPairs) + kMvrPairs];
int g_allPairCount = 0;

// Builder patch: three eye loop iterations instead of one, and the group array moved to free stack slots
// ([rsp+0x58] is a loop temporary, so the original array has room for one group only).
struct PatchByte {
    Anchor anchor;
    size_t off;
    uint8_t from, to;
};
const PatchByte kPatch[] = {
    {Anchor::BuilderLoopCmp, 2, 0x03, 0x05},        // cmp esi, 3 -> 5: views 2, 3, 4
    {Anchor::BuilderGroupStore, 4, 0x50, 0x60},     // mov [rsp+rax*8+0x50], r13
    {Anchor::BuilderGroupLoad, 4, 0x50, 0x60},      // mov rdx, [rsp+rax*8+0x50] (previous group)
    {Anchor::BuilderPostmainArray, 4, 0x50, 0x60},  // lea rbx, [rsp+0x50] (postmain depends on every group)
};

std::atomic<bool> g_on{false};
std::atomic<bool> g_fault{false};
bool g_installed = false;
bool g_patched = false;
uint8_t* g_app = nullptr;  // the engine's app singleton (static object, address taken from its getter)

// Quad <-> outer only (M4.5): the executor call sites go through RunTask, which skips every task of the focus groups
// while g_skip is set. Both Submit textures then hold outer (focus R no longer overwrites the right one).
// The tasks are a sorted, immutable array published after the builder; replaced arrays are freed at VR_Shutdown.
struct SkipSet {
    std::vector<uintptr_t> tasks;  // sorted
};
bool g_execPatched = false;
std::atomic<bool> g_skip{false};
std::atomic<const SkipSet*> g_skipSet{nullptr};  // null = unknown: nothing is skipped
std::vector<const SkipSet*> g_retired;
std::atomic<uint64_t> g_skipped{0};              // tasks not executed
struct AddCall {
    void* group;
    void* task;
};
std::vector<AddCall> g_adds;  // "add task" calls of the current builder call

// Runs f under SEH. A fault stops the focus swaps for the session (Safe skips its work from then on); the detours keep
// calling the originals. With the executor patch the focus groups are skipped from then on (outer only), and the next
// builder call goes without the builder patch; without it the focus groups render the right eye like outer R.
template <class F>
void Guard(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (g_execPatched) g_skip = true;
        if (!g_fault.exchange(true))
            Log("ERROR: exception 0x%08lX in quad mode (%s). Focus swaps stopped for this session; %s", code, where,
                g_execPatched ? "outer only (focus groups skipped, no builder patch at the next track load)"
                              : "the focus groups render the right eye");
    }
}

template <class F>
void Safe(const char* where, F&& f) {
    if (!g_fault.load(std::memory_order_relaxed)) Guard(where, f);
}

// ---------------------------------------------------------------- state

struct Graph {
    void* begin[kGroups] = {};  // CBeginEye tasks in build order
    void* end[kGroups] = {};    // CEndEye tasks
    int begins = 0, ends = 0, bonnets = 0;
    bool unexpected = false;    // view numbers out of order or too many groups
    bool ok = false;            // four complete groups: focus swaps on
};

struct Swap {
    uint8_t* ctx = nullptr;  // null = no swap active
    int group = -1;
    uint8_t slot[kSlotSize];
    uint8_t block[kCamBlockSize];
    int32_t query = 0;
    uint8_t* vr = nullptr;  // VR device whose view 2 eye data is swapped (null = not swapped)
    uint8_t eye[kVrEyeSize];
    uint8_t* cam = nullptr;  // camera whose projection is swapped (null = not swapped)
    uint8_t camProj[kMatSize];
};

struct Run {
    double t0 = 0, t1 = 0;
    void* out = nullptr;  // texture the group's CEndEye wrote (only compared)
    bool swapped = false;     // view 2 fields taken from the source view (focus L)
    bool eyeSwapped = false;  // VR device eye data of view 2 replaced
    bool projected = false;   // focus projection in the camera and in the eye data
    uint32_t x0 = 0, y0 = 0;  // with `projected`: the focus window it rendered (grid pixels), for the composition
    bool culled = false;      // the draw queue was culled for the focus window (M9.3)
    focus_cull::Result cull;
};

struct Stats {
    uint64_t frames = 0, full = 0, partial = 0, swaps = 0, eyeSwaps = 0, lateRestores = 0, outNotSubmit = 0;
    uint64_t projected = 0;   // frames with the focus projection in both focus groups
    uint64_t composed[2] = {};   // Submit per eye with the composed image
    uint64_t outerOnly[2] = {};  // without the composition: outer at the focus size (left: game texture, right: copy)
    uint64_t rightAsIs = 0;      // right eye: game texture (focus R), no outer R copy in this frame
    uint64_t outerFrames = 0;         // frames with the focus groups skipped
    uint64_t skipLeaks = 0;      // outer-only frames in which a focus group still ran (executor path not covered)
    uint64_t bySource[4] = {};   // frames with the focus groups by window source (live, held, fixed, frozen)
    uint64_t gazeLost = 0;       // the window went to the fixed center: no valid gaze for longer than gaze_hold_s
    double cpuMs[kGroups] = {};
    // Focus culling (M9.3), sums over the frames with both focus groups culled.
    uint64_t cullFrames = 0, cullOnFrames = 0;  // culled / with culling on and the focus projection in both groups
    double cullMs[2] = {};                      // gather + processor + queue copy per focus group
    double setEye = 0, setFocus[2] = {}, queueEye = 0, queueFocus[2] = {};
};

// Focus window for this VR session (M4.3, M7.2): the game gets the render size and the full projection (outer), the
// focus groups get the focus window of their eye. The window (geo.focusW x geo.focusH grid pixels) is the DFR or the FFR
// section, by follow_gaze at the start of the session. The game renders at the larger of the outer and the focus texture
// sizes per axis; a set whose size differs renders into clones in its own size (M7.4), or at the game size without them.
struct Focus {
    bool setupDone = false;
    bool on = false;          // geometry ready: the game got the render size
    bool engaged = false;     // the size was replaced in this session (the wrappers must stay)
    bool dfr = true;          // the window has the DFR section (else the FFR one)
    focus::Geometry geo;
    uint32_t outerW = 0, outerH = 0;    // outer texture wanted (peripheral_multiplier)
    uint32_t focusTexW = 0, focusTexH = 0;  // focus texture wanted (focus_multiplier x window)
    uint32_t gameW = 0, gameH = 0;      // size returned to the game = size of all four views without clones
    uint32_t outerOutW = 0, outerOutH = 0;  // outer L's output in the latest frame (with clones: outer's size)
    bool projFailed = false;  // engine data did not match the geometry: focus groups render the eye views
    int projMisses = 0;       // consecutive focus groups without the projection (projFailed at kProjMissLimit)
    bool missLogged = false;
    bool noCopyLogged = false;
    bool logged[kGroups] = {};
};
// About one second of frames with both focus groups failing the check; one-off misses (a camera rebuilding its
// projection right after the builder) only cost that frame's composition.
constexpr int kProjMissLimit = 144;

std::mutex g_m;
Graph g_graph;
int g_builderCalls = 0;
Swap g_swap;
Run g_run[kGroups];
unsigned g_began = 0, g_ended = 0;  // bit per group in the current frame
uint64_t g_frameNo = 0;
double g_frameT0 = 0;
double g_firstFocusMs = -1;
int g_detailLeft = 3;
bool g_outLogged[kGroups] = {};
Stats g_st, g_tot;
bool g_shotFrame = false;
bool g_autoShotDone = false;
bool g_keyWasDown = false;
void* g_lastSubmit[2] = {};  // texture of the latest Submit per eye (only compared)
Focus g_focus;
// Outer R, focus L and focus R all end in the right Submit texture, so the first two are copied at their CEndEye.
// The composition takes outer L from the left Submit texture and focus R from the right one.
struct Copy {
    ID3D11Texture2D* tex = nullptr;
    bool frame = false;  // holds this frame's image
    bool failed = false;
    bool sizeLogged = false;
};
enum { CopyOuterR, CopyFocusL, kCopies };
const char* const kCopyNames[kCopies] = {"outer R", "focus L"};
Copy g_copy[kCopies];
bool g_rightLogged = false;
bool g_cloneSubmitLogged = false;
char g_noCompose[2][160] = {};  // last reason logged per eye
uint32_t g_gen = 0;             // gpu::Generation() the D3D11 objects above belong to

// Outer only: requested by the hotkey or by the watchdog; g_frameSkip is the state of the frame being recorded.
bool g_frameSkip = false;
bool g_userOuter = false;
char g_autoOuter[192] = "";  // watchdog reason (empty = none); cleared at the next builder call
int g_noComposeRun = 0;       // frames in a row with the focus groups running and no eye composed
bool g_skipLeakLogged = false;
bool g_outerKeyDown = false, g_overlayKeyDown = false, g_freezeKeyDown = false;
// Clones of the eye-size targets for the focus groups (M7.3): setting and hotkey.
bool g_clonesOn = true;
bool g_clonesCfg = true;  // Cfg().clones last seen (a reload sets g_clonesOn)
bool g_clonesKeyDown = false;
// Focus culling (M9.3): setting and hotkey; checks that fail in a row turn it off until the next builder call, an engine
// fault for the session.
bool g_cullOn = true;
bool g_cullCfg = true;
bool g_cullKeyDown = false;
bool g_mvrOn = true;  // MVR gate B8 (the hotkey switches; mvr_gbuffer=1 installs it)
bool g_mvrKeyDown = false;
bool g_mvrModeKeyDown = false;
int g_mvrEarlyCfg = 0;  // Cfg().mvrEarly last seen (a reload sets the step)
bool g_mvrEarlyKeyDown = false;
bool g_mvrForwardCfg = false;  // Cfg().mvrForward last seen (a reload sets it)
bool g_mvrForwardKeyDown = false;
int g_mvrFwdDepthCfg = 3;  // Cfg().mvrForwardDepth last seen
bool g_mvrFwdDepthKeyDown = false;
bool g_mvrMapKeyDown = false;
bool g_mapFrame = false;  // a task map with PS resources in this frame (mvr_map_key)
bool g_mapNext = false;
// MVR forward step 1 (B9a): each eye group's CSSAORender and CPhase2Render (null when the graph has none or two).
struct EyeTasks {
    uint8_t* ssao = nullptr;
    uint8_t* phase2 = nullptr;
};
EyeTasks g_eyeTasks[kGroups];  // under g_m, from BuildProfileLocked
// Crepuscular rays and lens flares of the focus views from outer (sun_fx.h): setting and hotkey.
bool g_sunOn = true;
bool g_sunCfg = true;
bool g_sunKeyDown = false;
// Own TAA state of the focus views (taa_views.h): setting and hotkey.
bool g_taaOn = true;
bool g_taaCfg = true;
bool g_taaKeyDown = false;
struct Cull {
    bool failed = false;  // off until the next builder call
    int misses = 0;       // focus groups in a row without culling (checks failed)
    bool missLogged = false;
    bool logged = false;  // first culled frame of this graph logged
};
Cull g_cull;
bool g_cullFault = false;
constexpr int kCullMissLimit = 144;
constexpr float kCullMargin = 0.05f;  // focus planes pushed out by this much NDC per edge
alignas(16) uint8_t g_cullBlock[2][focus_cull::kBlockSize];  // the processor keeps a pointer until the next frame

// Debug overlay and gaze (NDC per eye from GetEyeTrackedFoveationCenter, x right, y up).
bool g_overlay = false;
bool g_overlayCfg = false;  // Cfg().overlay last seen (a reload sets g_overlay)
bool g_gazeOk = false, g_gazeFrozen = false;
float g_gaze[2][2] = {};

// DFR (M5): the focus window of each frame, placed at WaitGetPoses exit (SetGaze), before the frame's eye groups run.
// Like the mbucchia layer: on the gaze every frame, without smoothing or prediction; an invalid gaze keeps the last
// valid one for gaze_hold_s, then the window goes to the fixed center. A gaze that keeps its value (the tracker lost
// the eye, SteamVR still reports it valid) moves nothing and is only counted in frame.cpp.
struct Window {
    uint64_t frame = 0;  // g_frameNo it was placed for (0 = none)
    int src = SrcFixed;
    focus::Eye eye[2];   // frustum and focus window per eye
};
Window g_window;
bool g_follow = true;
bool g_followCfg = true;     // Cfg().followGaze last seen (a reload sets g_follow)
bool g_gazeKeyDown = false;
bool g_haveGood = false;     // a valid gaze was read in this session
double g_goodMs = 0;         // when it was read
float g_good[2][2] = {};
bool g_lost = false;         // following, but the window is at the center: no valid gaze for gaze_hold_s
int g_lostLogs = 0;          // loss and return lines logged (the rest only counted)
bool g_benchWindow = false;  // the previous frame was a benchmark frame
constexpr int kLostLogs = 10;

thread_local bool t_inBuilder = false;
thread_local int t_buildGroup = -1;
thread_local bool t_inEndEye = false;
thread_local ID3D11Resource* t_endRes = nullptr;  // referenced: the RTV may be gone once CEndEye returns

// D3D11: the game's immediate context comes from gpu::Context() (referenced there and checked at every Submit).
bool g_d3dHooked = false;

int Find(void* const* tasks, void* task) {
    for (int g = 0; g < kGroups; ++g)
        if (tasks[g] && tasks[g] == task) return g;
    return -1;
}

std::string DescribeTex(void* p) {
    char buf[128];
    ID3D11Texture2D* t = nullptr;
    if (p && SUCCEEDED(static_cast<IUnknown*>(p)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        t->Release();
        snprintf(buf, sizeof(buf), "%p %ux%u fmt %d", p, d.Width, d.Height, (int)d.Format);
    } else {
        snprintf(buf, sizeof(buf), "%p (not a 2D texture)", p);
    }
    return buf;
}

// ---------------------------------------------------------------- builder patch

bool WriteByte(uint8_t* a, uint8_t v) {
    DWORD old = 0;
    if (!VirtualProtect(a, 1, PAGE_EXECUTE_READWRITE, &old)) return false;
    *a = v;
    VirtualProtect(a, 1, old, &old);
    FlushInstructionCache(GetCurrentProcess(), a, 1);
    return true;
}

// Bytes already in the target state count as done, so a second call is harmless. A failed write puts the bytes
// written so far back: a half patch (the store at 0x60, the load at 0x50) would break the eye loop.
bool WritePatch(bool apply) {
    constexpr size_t n = std::size(kPatch);
    for (const PatchByte& p : kPatch) {
        uint8_t* a = (uint8_t*)hooks::Address(p.anchor);
        uint8_t from = apply ? p.from : p.to, to = apply ? p.to : p.from;
        if (!a || (a[p.off] != from && a[p.off] != to)) {
            Log("Quad: builder patch not %s: %s+%zu is 0x%02X, expected 0x%02X", apply ? "applied" : "removed",
                hooks::Name(p.anchor), p.off, a ? a[p.off] : 0, from);
            return false;
        }
    }
    bool written[n] = {};
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        const PatchByte& p = kPatch[i];
        uint8_t* a = (uint8_t*)hooks::Address(p.anchor) + p.off;
        uint8_t to = apply ? p.to : p.from;
        if (*a == to) continue;
        if (!WriteByte(a, to)) {
            Log("Quad: VirtualProtect failed at %s (error %lu); builder patch left %s", hooks::Name(p.anchor),
                GetLastError(), apply ? "off" : "on");
            for (size_t k = 0; k < i; ++k) {
                const PatchByte& q = kPatch[k];
                if (written[k]) WriteByte((uint8_t*)hooks::Address(q.anchor) + q.off, apply ? q.from : q.to);
            }
            return false;
        }
        written[i] = true;
        ++count;
    }
    Log("Quad: builder patch %s (4 bytes: eye loop for views 2..4, group array at [rsp+0x60])%s",
        apply ? "applied" : "removed", count ? "" : ", already in place");
    return true;
}

// ---------------------------------------------------------------- executor patch (quad <-> outer only)

using ExecuteFn = void(__fastcall*)(void*);

// True when the task belongs to a focus group (binary search, no game memory read).
bool InSkipSet(uint8_t* task) {
    const SkipSet* s = g_skipSet.load(std::memory_order_acquire);
    return s && std::binary_search(s->tasks.begin(), s->tasks.end(), (uintptr_t)task);
}

// Every task Execute of the game goes through here (the thunk replaces `mov rcx, rbx; call [rax+0x48]`). The executor
// still moves a skipped task through its states and releases its dependents, so the graph completes as usual.
// For the stall watch: the last task run and the last one skipped while outer only is on.
std::atomic<uint8_t*> g_lastRun{nullptr}, g_lastSkipped{nullptr};

// Task CPU profile (M9 step 4, [debug] task_profile): the Execute time of every leaf task of the four eye groups, from
// the builder's "add task" calls like the skip set. An immutable table per graph; counters are atomics, so the log
// thread can read and reset them while the render thread runs tasks. CBeginEye and CEndEye include the proxy's work in
// their detours (view swap, focus culling, copies).
struct ProfTask {
    int group = -1;
    int order = 0;  // build order (its add-task call)
    bool begin = false;  // the group's CBeginEye: its calls are the group's frames
    char name[48] = "";
    mutable std::atomic<uint64_t> ticks{0}, calls{0};
    mutable std::atomic<uint64_t> gapTicks{0};  // render thread: from the end of its previous task to this one's start
    mutable std::atomic<uint64_t> offThread{0};  // runs on another thread than the render thread
};
struct Profile {
    std::vector<uintptr_t> tasks;  // sorted
    std::vector<uint32_t> entry;   // entry of tasks[i]
    std::unique_ptr<ProfTask[]> entries;
    size_t n = 0;
};
std::atomic<const Profile*> g_profile{nullptr};
std::vector<const Profile*> g_profRetired;
std::atomic<bool> g_profOn{false};
std::atomic<DWORD> g_renderTid{0};  // the thread that runs CBeginEye
thread_local int64_t t_lastEnd = 0;  // end of the latest task this thread ran (with the profile on)
int64_t g_qpcFreq = 0;

const ProfTask* ProfFind(uint8_t* task) {
    const Profile* p = g_profile.load(std::memory_order_acquire);
    if (!p) return nullptr;
    auto it = std::lower_bound(p->tasks.begin(), p->tasks.end(), (uintptr_t)task);
    if (it == p->tasks.end() || *it != (uintptr_t)task) return nullptr;
    return &p->entries[p->entry[it - p->tasks.begin()]];
}

void EarlyFocus(int eye);

void __fastcall RunTask(uint8_t* task) {
    if (g_skip.load(std::memory_order_relaxed)) {
        if (InSkipSet(task)) {
            g_skipped.fetch_add(1, std::memory_order_relaxed);
            g_lastSkipped.store(task, std::memory_order_relaxed);
            (*(ExecuteFn**)task)[kFinishSlot](task);  // what the skipped Execute would have ended with
            return;
        }
        g_lastRun.store(task, std::memory_order_relaxed);
    }
    const bool profOn = g_profOn.load(std::memory_order_relaxed);
    const bool mvr = mvrpass::On();
    const ProfTask* prof = profOn || mvr ? ProfFind(task) : nullptr;
    // MVR forward step 1 (B9a): focus L's SSAO and lights right before outer L's forward pass; with the live forward
    // pass (B9d) focus R's before outer R's too (EarlyWanted decides).
    if (mvr && prof && (prof->group == OuterL || prof->group == OuterR) &&
        mvrpass::KindOf(task) == mvrpass::Kind::Phase3)
        EarlyFocus(prof->group - OuterL);
    // MVR gate B8: a focus group's G-buffer tasks are skipped when outer's pass drew its G-buffer (B9a: and its SSAO
    // and lights after an early run).
    if (mvr && mvrpass::BeforeTask(task, prof ? prof->group : -1)) {
        (*(ExecuteFn**)task)[kFinishSlot](task);
        return;
    }
    // Task map of a shot frame: what each task of the eye groups draws and binds.
    const bool map = mvr && prof && (g_shotFrame || g_mapFrame) &&
                     GetCurrentThreadId() == g_renderTid.load(std::memory_order_relaxed);
    if (map) mvrpass::TaskMapBegin(kGroupNames[prof->group], prof->name, g_mapFrame);
    LARGE_INTEGER t0{}, t1{};
    if (profOn) QueryPerformanceCounter(&t0);
    (*(ExecuteFn**)task)[kExecuteSlot](task);
    if (map) mvrpass::TaskMapEnd();
    if (mvr) mvrpass::AfterTask(task);
    if (profOn) {
        QueryPerformanceCounter(&t1);
        if (prof) {
            prof->ticks.fetch_add((uint64_t)(t1.QuadPart - t0.QuadPart), std::memory_order_relaxed);
            prof->calls.fetch_add(1, std::memory_order_relaxed);
            if (GetCurrentThreadId() != g_renderTid.load(std::memory_order_relaxed)) {
                prof->offThread.fetch_add(1, std::memory_order_relaxed);
            } else if (t_lastEnd) {
                // A gap longer than 20 ms is a frame break (menu, loading), not work between two tasks.
                const int64_t gap = t0.QuadPart - t_lastEnd;
                if (gap > 0 && gap < g_qpcFreq / 50) prof->gapTicks.fetch_add((uint64_t)gap, std::memory_order_relaxed);
            }
        }
        t_lastEnd = t1.QuadPart;  // nested tasks first, then their group: the group's end wins
    }
}

const Anchor kExecSites[] = {Anchor::ExecCallA, Anchor::ExecCallB};
const uint8_t kExecOrig[6] = {0x48, 0x8B, 0xCB, 0xFF, 0x50, 0x48};  // mov rcx, rbx; call [rax+0x48]
uint8_t* g_thunk = nullptr;

// The thunk is called with rbx = task and a 16-byte aligned stack minus the return address:
//   sub rsp, 0x28; mov rcx, rbx; mov rax, RunTask; call rax; add rsp, 0x28; ret
// It lives within rel32 reach of the exe, with unwind data, so exceptions and stack walks pass through it.
uint8_t* MakeThunk(uintptr_t around) {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uintptr_t gran = si.dwAllocationGranularity;
    uint8_t* page = nullptr;
    for (uintptr_t d = gran; d < 0x7000'0000 && !page; d += gran) {
        for (int s = 0; s < 2 && !page; ++s) {
            if (!s && around < d) continue;
            uintptr_t a = (s ? around + d : around - d) & ~(gran - 1);
            page = (uint8_t*)VirtualAlloc((void*)a, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        }
    }
    if (!page) return nullptr;
    uint8_t code[24] = {0x48, 0x83, 0xEC, 0x28, 0x48, 0x8B, 0xCB, 0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0,
                        0xFF, 0xD0, 0x48, 0x83, 0xC4, 0x28, 0xC3};
    const uint64_t fn = (uint64_t)&RunTask;
    memcpy(code + 9, &fn, 8);
    memcpy(page, code, sizeof(code));
    // UNWIND_INFO at +32: version 1, prolog 4 bytes, one code: at +4 UWOP_ALLOC_SMALL (0x28 = (4 + 1) * 8).
    const uint8_t unwind[8] = {0x01, 0x04, 0x01, 0x00, 0x04, 0x42, 0x00, 0x00};
    memcpy(page + 32, unwind, sizeof(unwind));
    auto* rf = (RUNTIME_FUNCTION*)(page + 48);
    rf->BeginAddress = 0;
    rf->EndAddress = sizeof(code);
    rf->UnwindData = 32;
    DWORD old = 0;
    if (!VirtualProtect(page, 4096, PAGE_EXECUTE_READ, &old) || !RtlAddFunctionTable(rf, 1, (DWORD64)page)) {
        VirtualFree(page, 0, MEM_RELEASE);
        return nullptr;
    }
    FlushInstructionCache(GetCurrentProcess(), page, 4096);
    return page;
}

// Six bytes per site as one 8-byte store (both sites sit inside a cache line), so a thread running the executor sees
// either the old or the new instruction pair.
bool WriteSite(uint8_t* a, const uint8_t* bytes6) {
    DWORD old = 0;
    if (!VirtualProtect(a, 8, PAGE_EXECUTE_READWRITE, &old)) return false;
    uint64_t v = 0;
    memcpy(&v, a, 8);
    memcpy(&v, bytes6, 6);
    if (((uintptr_t)a & 63) <= 56)
        *(volatile uint64_t*)a = v;
    else
        memcpy(a, &v, 8);
    VirtualProtect(a, 8, old, &old);
    FlushInstructionCache(GetCurrentProcess(), a, 8);
    return true;
}

// Once per process; the patch stays (RunTask only forwards while nothing is skipped).
bool PatchExecutor() {
    uint8_t* site[std::size(kExecSites)] = {};
    for (size_t i = 0; i < std::size(kExecSites); ++i) {
        site[i] = (uint8_t*)hooks::Address(kExecSites[i]);
        if (!site[i] || memcmp(site[i], kExecOrig, 6) != 0) {
            Log("Quad: executor site %s does not match; no quad <-> outer only switch", hooks::Name(kExecSites[i]));
            return false;
        }
    }
    g_thunk = MakeThunk((uintptr_t)site[0]);
    if (!g_thunk) {
        Log("Quad: no memory for the executor thunk near the exe (error %lu); no quad <-> outer only switch",
            GetLastError());
        return false;
    }
    for (size_t i = 0; i < std::size(kExecSites); ++i) {
        const int64_t rel = (int64_t)g_thunk - (int64_t)(site[i] + 5);
        uint8_t call[6] = {0xE8, 0, 0, 0, 0, 0x90};
        const int32_t rel32 = (int32_t)rel;
        memcpy(call + 1, &rel32, 4);
        if (rel != rel32 || !WriteSite(site[i], call)) {
            for (size_t k = 0; k < i; ++k) WriteSite(site[k], kExecOrig);
            Log("Quad: executor site %s not patched (thunk %p out of reach or VirtualProtect error %lu); no quad <-> "
                "outer only switch", hooks::Name(kExecSites[i]), (void*)g_thunk, GetLastError());
            return false;
        }
    }
    Log("Quad: task executor patched (2 call sites -> filter thunk %p); quad <-> outer only available", (void*)g_thunk);
    return true;
}

void PublishSkipSet(const SkipSet* s) {
    if (const SkipSet* old = g_skipSet.exchange(s)) g_retired.push_back(old);  // a reader may still be in it
}

void PublishProfile(const Profile* p) {
    if (const Profile* old = g_profile.exchange(p)) g_profRetired.push_back(old);  // a reader may still be in it
}

void ClearGroups() {
    PublishSkipSet(nullptr);
    PublishProfile(nullptr);
    for (EyeTasks& e : g_eyeTasks) e = EyeTasks{};
}

using TaskNameFn = const char*(__fastcall*)(void*);

// The task's class name from vtable +0x30 (the executor calls it before Execute), under SEH.
bool TaskClassName(void* task, char* out, size_t n) {
    out[0] = 0;
    __try {
        const char* s = (*(TaskNameFn**)task)[0x30 / 8](task);
        if (!s) return false;
        size_t i = 0;
        for (; i + 1 < n && s[i]; ++i) out[i] = s[i];
        out[i] = 0;
        return i > 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        out[0] = 0;
        return false;
    }
}

// After a builder call: the leaf tasks of each eye group, from the groups its CBeginEye and CEndEye were added to and
// the groups inside them. Outer L's CBeginEye sits in preMain, so outer L also gets the once-per-frame tasks; a task in
// several closures goes to the most specific group (focus R, focus L, outer R, then outer L).
void BuildProfileLocked() {
    PublishProfile(nullptr);
    if (!g_execPatched || !g_graph.ok) return;
    auto groupOf = [](void* task) -> void* {
        for (const AddCall& a : g_adds)
            if (a.task == task) return a.group;
        return nullptr;
    };
    std::unordered_set<void*> member[kGroups];
    for (int g = 0; g < kGroups; ++g) {
        std::unordered_set<void*>& m = member[g];
        for (void* t : {g_graph.begin[g], g_graph.end[g]}) {
            m.insert(t);
            if (void* grp = groupOf(t)) m.insert(grp);
        }
        for (bool grew = true; grew;) {
            grew = false;
            for (const AddCall& a : g_adds)
                if (m.count(a.group) && m.insert(a.task).second) grew = true;
        }
    }
    auto* p = new Profile;
    std::unordered_set<void*> taken;
    struct Item {
        void* task;
        int group, order;
    };
    std::vector<Item> items;
    const int pick[kGroups] = {FocusR, FocusL, OuterR, OuterL};
    for (int g : pick)
        for (size_t i = 0; i < g_adds.size(); ++i) {
            void* t = g_adds[i].task;
            if (!member[g].count(t) || taken.count(t)) continue;
            uintptr_t vt = 0, slot = 0;
            if (!hooks::TryRead(t, &vt, 8) || !hooks::TryRead((void*)(vt + kIsGroupSlot * 8), &slot, 8) ||
                slot != hooks::ExeBase() + kLeafIsGroup)
                continue;  // groups: their Execute runs the children, which are timed themselves
            taken.insert(t);
            items.push_back({t, g, (int)i});
        }
    // CBeginEye and CEndEye always belong to their own group (outer L's CBeginEye also sits in preMain's closure).
    for (int g = 0; g < kGroups; ++g)
        for (void* t : {g_graph.begin[g], g_graph.end[g]})
            if (!taken.count(t)) {
                taken.insert(t);
                items.push_back({t, g, -1});
            }
    for (EyeTasks& e : g_eyeTasks) e = EyeTasks{};
    bool twice[kGroups] = {};
    for (const Item& it : items) {
        const mvrpass::Kind k = mvrpass::KindOf((uint8_t*)it.task);
        uint8_t** slot = k == mvrpass::Kind::Ssao ? &g_eyeTasks[it.group].ssao
                         : k == mvrpass::Kind::Phase2 ? &g_eyeTasks[it.group].phase2 : nullptr;
        if (!slot) continue;
        if (*slot) twice[it.group] = true;
        *slot = (uint8_t*)it.task;
    }
    for (int g = 0; g < kGroups; ++g)
        if (twice[g]) g_eyeTasks[g] = EyeTasks{};  // not one task of each: no early run for that group
    p->n = items.size();
    p->entries = std::make_unique<ProfTask[]>(p->n);
    size_t count[kGroups] = {};
    for (size_t i = 0; i < p->n; ++i) {
        ProfTask& e = p->entries[i];
        e.group = items[i].group;
        e.order = items[i].order;
        e.begin = items[i].task == g_graph.begin[e.group];
        if (!TaskClassName(items[i].task, e.name, sizeof(e.name)))
            snprintf(e.name, sizeof(e.name), "task@%p", items[i].task);
        ++count[e.group];
    }
    // MVR forward (B9d): the focus groups' CPhase3BuildCommandThread is skipped with their CPhase3Render (the pair
    // records and replays the group's buffers); its vtable by class name.
    uintptr_t build3 = 0;
    for (size_t i = 0; i < p->n && !build3; ++i)
        if (p->entries[i].group >= FocusL && !strcmp(p->entries[i].name, "CPhase3BuildCommandThread"))
            hooks::TryRead(items[i].task, &build3, 8);
    mvrpass::SetPhase3BuildVtable(build3);
    std::vector<uint32_t> idx(p->n);
    for (uint32_t i = 0; i < p->n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return items[a].task < items[b].task; });
    for (uint32_t i : idx) {
        p->tasks.push_back((uintptr_t)items[i].task);
        p->entry.push_back(i);
    }
    Log("Quad: task profile: %zu leaf tasks (outer L %zu, outer R %zu, focus L %zu, focus R %zu)%s", p->n, count[OuterL],
        count[OuterR], count[FocusL], count[FocusR], g_profOn ? "" : "; off ([debug] task_profile=0)");
    PublishProfile(p);
}

// One line per eye group: per frame and task in build order its Execute time, "+" the render thread's gap before it
// (from the end of the task it ran before; for CBeginEye that is the gap between groups) and "w" the share of its runs on
// other threads; then the sums against the group's CPU (CBeginEye to CEndEye). Counters are reset.
void LogProfileLocked(const Stats& s) {
    const Profile* p = g_profile.load(std::memory_order_acquire);
    if (!p) return;
    if (!g_qpcFreq) return;
    const double freq = (double)g_qpcFreq;
    std::vector<uint32_t> order(p->n);
    for (uint32_t i = 0; i < p->n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const ProfTask &x = p->entries[a], &y = p->entries[b];
        return x.group != y.group ? x.group < y.group : x.order < y.order;
    });
    std::vector<uint64_t> ticks(p->n), calls(p->n), gaps(p->n), off(p->n);
    for (uint32_t i = 0; i < p->n; ++i) {
        ticks[i] = p->entries[i].ticks.exchange(0, std::memory_order_relaxed);
        calls[i] = p->entries[i].calls.exchange(0, std::memory_order_relaxed);
        gaps[i] = p->entries[i].gapTicks.exchange(0, std::memory_order_relaxed);
        off[i] = p->entries[i].offThread.exchange(0, std::memory_order_relaxed);
    }
    if (!g_profOn) return;
    for (int g = 0; g < kGroups; ++g) {
        uint64_t frames = 0;
        for (uint32_t i = 0; i < p->n; ++i)
            if (p->entries[i].group == g && p->entries[i].begin) frames = calls[i];
        if (!frames) continue;
        std::string line;
        double sum = 0, gapSum = 0, offSum = 0;
        for (uint32_t i : order) {
            const ProfTask& e = p->entries[i];
            if (e.group != g || !calls[i]) continue;
            const double ms = ticks[i] * 1000.0 / freq / frames;
            const double gap = gaps[i] * 1000.0 / freq / frames;
            sum += ms;
            if (!e.begin) gapSum += gap;
            if (off[i]) offSum += ms;
            if (ms < 0.0005 && gap < 0.0005) continue;  // keeps the line short (outer L has about 70 tasks)
            char item[128];
            int k = snprintf(item, sizeof(item), "%s %.3f", e.name, ms);
            if (gap >= 0.0005) k += snprintf(item + k, sizeof(item) - k, "+%.3f", gap);
            if (calls[i] != frames) k += snprintf(item + k, sizeof(item) - k, " x%.2f", calls[i] / (double)frames);
            if (off[i]) snprintf(item + k, sizeof(item) - k, " w%.0f%%", 100.0 * off[i] / calls[i]);
            if (!line.empty()) line += ", ";
            line += item;
        }
        const double group = s.full ? s.cpuMs[g] / s.full : 0;
        Log("Quad task CPU per frame (%s, %llu frames; group %.3f ms = tasks %.3f + rest %.3f; gaps inside %.3f, tasks on "
            "other threads %.3f): %s", kGroupNames[g], (unsigned long long)frames, group, sum, group - sum, gapSum,
            offSum, line.c_str());
    }
}

// After a builder call: every task the builder added to a focus group, and to the groups inside it (from the recorded
// "add task" calls). The group of each eye group is the one its CBeginEye was added to; the four must differ, each
// focus set must hold its CBeginEye and CEndEye and no outer task, else nothing is skipped (outer only off for this graph).
void BuildSkipSetLocked() {
    ClearGroups();
    if (!g_execPatched || !g_graph.ok) return;
    void* grp[kGroups] = {};
    for (const AddCall& a : g_adds)
        for (int g = 0; g < kGroups; ++g)
            if (!grp[g] && a.task == g_graph.begin[g]) grp[g] = a.group;
    // Diagnostics: the build-time group and +0x848 of CBeginEye/CEndEye after the builder.
    char now[kGroups][64] = {};
    for (int g = 0; g < kGroups; ++g) {
        void* b = nullptr;
        void* e = nullptr;
        hooks::TryRead((uint8_t*)g_graph.begin[g] + kTaskGroup, &b, 8);
        hooks::TryRead((uint8_t*)g_graph.end[g] + kTaskGroup, &e, 8);
        snprintf(now[g], sizeof(now[g]), "%p/%p", b, e);
    }
    char why[192] = "";
    for (int g = 0; g < kGroups && !why[0]; ++g) {
        if (!grp[g]) snprintf(why, sizeof(why), "%s: CBeginEye not added to a group in the builder", kGroupNames[g]);
        for (int k = 0; k < g && !why[0]; ++k)
            if (grp[k] == grp[g])
                snprintf(why, sizeof(why), "%s and %s share group %p", kGroupNames[k], kGroupNames[g], grp[g]);
    }
    std::unordered_set<void*> member[2];
    for (int f = 0; f < 2 && !why[0]; ++f) {
        std::unordered_set<void*>& m = member[f];
        m.insert(grp[FocusL + f]);
        // Closure over nested groups; a group may get its tasks before or after it is added itself.
        for (bool grew = true; grew;) {
            grew = false;
            for (const AddCall& a : g_adds)
                if (m.count(a.group) && m.insert(a.task).second) grew = true;
        }
        if (!m.count(g_graph.begin[FocusL + f]) || !m.count(g_graph.end[FocusL + f]))
            snprintf(why, sizeof(why), "%s set of %zu tasks misses its CBeginEye or CEndEye", kGroupNames[FocusL + f],
                     m.size());
    }
    for (int f = 0; f < 2 && !why[0]; ++f)
        for (int g = OuterL; g <= OuterR; ++g)
            if (member[f].count(g_graph.begin[g]) || member[f].count(g_graph.end[g]) || member[f].count(grp[g]))
                snprintf(why, sizeof(why), "%s set holds tasks of %s", kGroupNames[FocusL + f], kGroupNames[g]);
    if (!why[0])
        for (void* t : member[0])
            if (member[1].count(t)) {
                snprintf(why, sizeof(why), "focus L and focus R sets share task %p", t);
                break;
            }
    Log("Quad: builder recorded %zu add-task calls; eye task groups (build time) outer L %p, outer R %p, focus L %p, "
        "focus R %p; +0x848 of CBeginEye/CEndEye now: %s, %s, %s, %s",
        g_adds.size(), grp[0], grp[1], grp[2], grp[3], now[0], now[1], now[2], now[3]);
    if (why[0]) {
        Log("Quad: focus tasks not recognized (%s); quad <-> outer only off for this graph", why);
        return;
    }
    // Only tasks added to nothing but the focus sets are skipped: a task (or the focus group itself) that was also
    // added to a group outside them keeps running.
    std::unordered_set<void*> shared;
    for (const AddCall& a : g_adds) {
        const bool in = member[0].count(a.task) || member[1].count(a.task);
        if (in && !member[0].count(a.group) && !member[1].count(a.group)) shared.insert(a.task);
    }
    for (int f = 0; f < 2 && !why[0]; ++f)
        if (shared.count(g_graph.begin[FocusL + f]) || shared.count(g_graph.end[FocusL + f]))
            snprintf(why, sizeof(why), "%s CBeginEye or CEndEye is also in another group", kGroupNames[FocusL + f]);
    if (why[0]) {
        Log("Quad: focus tasks not recognized (%s); quad <-> outer only off for this graph", why);
        return;
    }
    // Only leaf tasks are skipped. A group's Execute runs its children's scheduler and the executor waits for its
    // result (+0xf50), so a skipped group never finishes and the frame hangs (second M4.5 run). Leaves are the tasks
    // whose vtable +0x28 is the shared "return false" (the executor's second path).
    auto* s = new SkipSet;
    size_t groups = 0;
    for (const auto& m : member)
        for (void* t : m) {
            if (shared.count(t)) continue;
            uintptr_t vt = 0, slot = 0, finish = 0;
            if (!hooks::TryRead(t, &vt, 8) || !hooks::TryRead((void*)(vt + kIsGroupSlot * 8), &slot, 8) ||
                !hooks::TryRead((void*)(vt + kFinishSlot * 8), &finish, 8) || slot != hooks::ExeBase() + kLeafIsGroup ||
                finish != hooks::ExeBase() + kTaskFinish) {
                ++groups;
                continue;
            }
            s->tasks.push_back((uintptr_t)t);
        }
    std::sort(s->tasks.begin(), s->tasks.end());
    for (int f = 0; f < 2 && !why[0]; ++f)
        if (!std::binary_search(s->tasks.begin(), s->tasks.end(), (uintptr_t)g_graph.begin[FocusL + f]))
            snprintf(why, sizeof(why), "%s CBeginEye is not a leaf task", kGroupNames[FocusL + f]);
    if (why[0]) {
        delete s;
        Log("Quad: focus tasks not recognized (%s); quad <-> outer only off for this graph", why);
        return;
    }
    Log("Quad: outer-only mode skips %zu leaf tasks (Execute replaced by Finish; focus L %zu, focus R %zu tasks; %zu also in "
        "other groups and %zu groups or other tasks keep running)",
        s->tasks.size(), member[0].size(), member[1].size(), shared.size(), groups);
    PublishSkipSet(s);
}

// The recorded graph is still alive: the focus groups' CBeginEye and CEndEye keep their vtables.
bool GroupsAliveLocked() {
    for (int f = 0; f < 2; ++f) {
        uintptr_t vt[2] = {};
        if (!hooks::TryRead(g_graph.begin[FocusL + f], &vt[0], 8) || !hooks::TryRead(g_graph.end[FocusL + f], &vt[1], 8) ||
            vt[0] != hooks::ExeBase() + kBeginEyeVtable || vt[1] != hooks::ExeBase() + kEndEyeVtable)
            return false;
    }
    return true;
}

void FreeRetiredSets() {
    for (const SkipSet* s : g_retired) delete s;
    g_retired.clear();
    for (const Profile* p : g_profRetired) delete p;
    g_profRetired.clear();
}

// Stall watch (its own thread, never touches the render thread's state): when no frame starts for 3 s while outer only
// is on, the log gets the last task run and the last one skipped, with their vtables, once per stall.
std::atomic<double> g_frameStartMs{0};

std::string TaskName(uint8_t* t) {
    char buf[96];
    uintptr_t vt = 0;
    if (!t || !hooks::TryRead(t, &vt, 8))
        snprintf(buf, sizeof(buf), "%p", (void*)t);
    else
        snprintf(buf, sizeof(buf), "%p (vtable AMS2AVX.exe+0x%llX)", (void*)t, (unsigned long long)(vt - hooks::ExeBase()));
    return buf;
}

void StartStallWatch() {
    std::thread([] {
        bool logged = false;
        for (;;) {
            Sleep(500);
            const double since = NowMs() - g_frameStartMs.load();
            if (!g_skip.load() || g_frameStartMs.load() == 0 || since < 3000) {
                logged = false;
                continue;
            }
            if (logged) continue;
            logged = true;
            Guarded("stall watch", [&] {
                Log("Quad: STALL: no frame for %.1f s in outer-only mode; last task run %s, last skipped %s, %llu skipped so far",
                    since / 1000.0, TaskName(g_lastRun.load()).c_str(), TaskName(g_lastSkipped.load()).c_str(),
                    (unsigned long long)g_skipped.load());
            });
        }
    }).detach();
}

// ---------------------------------------------------------------- view 2 swap

// The engine's VR device (app+0xe0), or null outside VR. The app type check is the one the engine does before it.
uint8_t* VrDevice() {
    if (!g_app) return nullptr;
    int type = *(int*)(g_app + kAppType);
    if (type != 5 && (unsigned)(type - 7) >= 3) return nullptr;
    return *(uint8_t**)(g_app + kAppVr);
}

bool NearTan(float a, float b) { return std::fabs(a - b) <= 2e-3f; }

// Focus eye data from the source eye's: the same block with the focus window's tangents and projection. The engine
// builds the matrix from the tangents the game got from GetProjectionMatrix (full FOV in quad mode), so both must match
// the geometry; otherwise false (and the reason in `why`).
bool FocusEyeData(const uint8_t* src, const focus::Eye& e, uint8_t* out, char* why, size_t whySize) {
    memcpy(out, src, kVrEyeSize);
    const float* tan = (const float*)(src + kEyeTangents);
    const float* m = (const float*)(src + kEyeProj);  // m[i * 4 + j]: row i, column j
    if (!NearTan(tan[0], e.b) || !NearTan(tan[1], -e.t) || !NearTan(tan[2], -e.l) || !NearTan(tan[3], e.r)) {
        snprintf(why, whySize, "engine tangents {%.5f %.5f %.5f %.5f}, expected {b -t -l r} = {%.5f %.5f %.5f %.5f}",
                 tan[0], tan[1], tan[2], tan[3], e.b, -e.t, -e.l, e.r);
        return false;
    }
    // Clip space of the focus window: x' = sx * x + ox * w, y' = sy * y + oy * w (NDC x = m00 * x/z + m20 for the
    // engine's z forward, the same function of the view tangent as in OpenVR). Row vectors, so the x and y columns
    // of the matrix change.
    const double sx = (e.r - e.l) / (double)(e.fr - e.fl), ox = ((e.r + e.l) - (e.fr + e.fl)) / (double)(e.fr - e.fl);
    const double sy = (e.b - e.t) / (double)(e.fb - e.ft), oy = ((e.b + e.t) - (e.fb + e.ft)) / (double)(e.fb - e.ft);
    float* fm = (float*)(out + kEyeProj);
    for (int i = 0; i < 4; ++i) {
        fm[i * 4 + 0] = (float)(sx * m[i * 4 + 0] + ox * m[i * 4 + 3]);
        fm[i * 4 + 1] = (float)(sy * m[i * 4 + 1] + oy * m[i * 4 + 3]);
    }
    float* ft = (float*)(out + kEyeTangents);
    ft[0] = e.fb;
    ft[1] = -e.ft;
    ft[2] = -e.fl;
    ft[3] = e.fr;
    // The engine's own tangent formula (0xd6bf30) on the new matrix must give the focus window back.
    float chk[4] = {(1 - fm[9]) / fm[5], (1 + fm[9]) / fm[5], (1 + fm[8]) / fm[0], (1 - fm[8]) / fm[0]};
    for (int k = 0; k < 4; ++k) {
        if (!NearTan(chk[k], ft[k])) {
            snprintf(why, whySize, "focus matrix gives tangents {%.5f %.5f %.5f %.5f}, expected {%.5f %.5f %.5f %.5f}",
                     chk[0], chk[1], chk[2], chk[3], ft[0], ft[1], ft[2], ft[3]);
            return false;
        }
    }
    return true;
}

// The camera of the view 2 slot gets the focus projection; it must be an engine camera that holds exactly the
// source eye's matrix (the camera functor sets it every frame) and no pending rebuild from FOV.
bool CameraReady(const uint8_t* cam, const uint8_t* srcEye, char* why, size_t whySize) {
    if (!cam) {
        snprintf(why, whySize, "no camera in the view 2 slot");
        return false;
    }
    uintptr_t vt = *(const uintptr_t*)cam;
    if (vt != hooks::ExeBase() + kCamVtable) {
        snprintf(why, whySize, "camera %p has vtable AMS2AVX.exe+0x%llX, expected +0x%llX", (const void*)cam,
                 (unsigned long long)(vt - hooks::ExeBase()), (unsigned long long)kCamVtable);
        return false;
    }
    if (cam[kCamProjDirty]) {
        snprintf(why, whySize, "camera %p rebuilds its projection from FOV (flag +0x402 set)", (const void*)cam);
        return false;
    }
    if (memcmp(cam + kCamProj, srcEye + kEyeProj, kMatSize) != 0) {
        const float* a = (const float*)(cam + kCamProj);
        const float* b = (const float*)(srcEye + kEyeProj);
        snprintf(why, whySize, "camera %p projection [%.5f %.5f %.5f %.5f] is not the eye's [%.5f %.5f %.5f %.5f]",
                 (const void*)cam, a[0], a[5], a[8], a[9], b[0], b[5], b[8], b[9]);
        return false;
    }
    return true;
}

void LogFocusFirst(int group, const uint8_t* src, const uint8_t* foc, const uint8_t* cam) {
    const float* t0 = (const float*)(src + kEyeTangents);
    const float* t1 = (const float*)(foc + kEyeTangents);
    const float* m0 = (const float*)(src + kEyeProj);
    const float* m1 = (const float*)(foc + kEyeProj);
    Log("Quad: %s projection: eye tangents {b -t -l r} {%.5f %.5f %.5f %.5f} -> {%.5f %.5f %.5f %.5f}; matrix m00 m11 "
        "m20 m21 [%.5f %.5f %.5f %.5f] -> [%.5f %.5f %.5f %.5f]; camera %p",
        kGroupNames[group], t0[0], t0[1], t0[2], t0[3], t1[0], t1[1], t1[2], t1[3], m0[0], m0[5], m0[8], m0[9], m1[0],
        m1[5], m1[8], m1[9], (const void*)cam);
}

// This frame's focus window of eye i: the one SetGaze placed, or the session's fixed one (SetupLocked) when SetGaze
// did not run for this frame.
const focus::Eye& WindowEyeLocked(int i) {
    return g_window.frame && g_window.frame == g_frameNo ? g_window.eye[i] : g_focus.geo.eye[i];
}

// Window center tangents of eye i (0 = left) as in the mbucchia layer: the gaze (NDC of the full projection, null =
// none) plus the focus offsets, or straight ahead (plus the pre-M7 center_x/center_y tangents) plus the fixed offsets.
// Offsets are NDC; the horizontal one is negated for the left eye, so + moves both windows away from the nose.
void WindowCenter(const focus::Eye& e, int i, const float* ndc, float& cx, float& cy) {
    const Config& c = Cfg();
    const float mirror = i ? 1.0f : -1.0f;
    if (ndc) {
        focus::NdcToTangent(e, ndc[0] + mirror * c.focusOffsetX, ndc[1] + c.focusOffsetY, cx, cy);
        return;
    }
    cx = c.quadCenterX + mirror * c.fixedOffsetX * 0.5f * (e.r - e.l);
    cy = c.quadCenterY + c.fixedOffsetY * 0.5f * (e.b - e.t);
}

// Grid pixels from the window edge to where the focus shows pixel for pixel (the larger axis with the mbucchia curve),
// left out of the focus vs composed shot comparison.
float SharpInsetLocked() {
    const Config& c = Cfg();
    if (c.legacyEdge) return (float)(c.quadMarginPx + c.quadFeatherPx);
    const float shown = (float)std::max(g_focus.geo.focusW, g_focus.geo.focusH) - 2.0f * c.quadMarginPx;
    return c.quadMarginPx + c.smoothen * std::max(0.0f, shown);
}

void CullMissLocked(int group, const char* why) {
    if (!g_cull.missLogged) {
        g_cull.missLogged = true;
        Log("Quad: %s without focus culling in frame %llu (%s); off after %d misses in a row", kGroupNames[group],
            (unsigned long long)g_frameNo, why, kCullMissLimit);
    }
    if (++g_cull.misses >= kCullMissLimit) {
        g_cull.failed = true;
        Log("Quad: focus culling OFF until the next builder call (%s: %s); the focus groups draw the eye's set",
            kGroupNames[group], why);
    }
}

// Focus culling (M9.3), after the focus projection is in: the engine's gather and processor run again for a frustum
// block with the focus window's side planes, and the draw queue the group reads gets their result. The block layout is
// learned from the source eye's engine block every time, so a block that does not match the expected format is never
// written. Near and far, the camera (distances and LOD) and the include and exclude flags stay the eye's.
void CullLocked(uint8_t* ctx, int group, const uint8_t* cam, const uint8_t* srcEye, const uint8_t* focusEye, Run& run) {
    if (!g_cullOn || g_cull.failed || g_cullFault || !focus_cull::Available()) return;
    char why[256] = "";
    if (cam[kCamRotDirty] || cam[kCamPosDirty]) {
        CullMissLocked(group, "camera view matrix not up to date (flags +0x400/+0x401)");
        return;
    }
    const float* view = (const float*)(cam + kCamView);
    const uint8_t* eyeBlock = ctx + kCtxCamBlock + kFocusView * kCamBlockSize;  // the source view's after the swap
    focus_cull::Layout lay;
    if (!focus_cull::Learn(eyeBlock, view, (const float*)(srcEye + kEyeProj), lay, why, sizeof(why))) {
        CullMissLocked(group, why);
        return;
    }
    uint8_t* blk = g_cullBlock[group - FocusL];
    focus_cull::Build(eyeBlock, lay, view, (const float*)(focusEye + kEyeProj), kCullMargin, blk);
    focus_cull::Result res;
    bool fault = false;
    if (!focus_cull::Run(ctx, blk, res, fault, why, sizeof(why))) {
        if (res.notYet) return;
        if (fault) {
            g_cullFault = true;
            Log("Quad: ERROR: focus culling stopped for this session (%s, frame %llu): %s", kGroupNames[group],
                (unsigned long long)g_frameNo, why);
        } else {
            CullMissLocked(group, why);
        }
        return;
    }
    g_cull.misses = 0;
    run.culled = true;
    run.cull = res;
    if (!g_cull.logged) {
        g_cull.logged = true;
        Log("Quad: %s culled for the focus window (frame %llu, %.3f ms): renderables %u -> %u, draw queue %u -> %u; "
            "block planes left/right/bottom/top %d %d %d %d, masks 0x%08X/0x%08X, margin %.2f NDC",
            kGroupNames[group], (unsigned long long)g_frameNo, res.ms, res.setEye, res.setFocus, res.queueEye,
            res.queueFocus, lay.side[0], lay.side[1], lay.side[2], lay.side[3], lay.maskSet, lay.maskClear, kCullMargin);
    }
}

// Before a focus group: view 2 takes the source view's camera slot, frustum block and lighting environment (focus L),
// and the camera and the VR device eye data of view 2 get the focus window of the group's eye. The deferred lighting
// reads the eye tangents straight from the VR device by view - 1 (0x9ce0e8), the passes read the projection from the
// camera (+0x138, +0x150). The frustum block stays the eye's (built from FOV, a superset of the focus window); focus
// culling gives the group a draw queue culled for its window.
// `early`: the MVR early run inside outer's group (B9a, B9d) renders no image of its own, so focus R needs no outer R
// copy yet (it is made at outer R's CEndEye, after the run); focus R's own group checks it as always.
void SwapInLocked(uint8_t* ctx, int group, int src, Run& run, bool cull = true, bool early = false) {
    Swap& s = g_swap;
    memcpy(s.slot, ctx + kCtxSlots + kFocusView * kSlotSize, kSlotSize);
    memcpy(s.block, ctx + kCtxCamBlock + kFocusView * kCamBlockSize, kCamBlockSize);
    memcpy(&s.query, ctx + kCtxQuery + kFocusView * 4, 4);
    // s.ctx, s.vr and s.cam are set once their originals are saved, so a fault halfway leaves RestoreLocked correct.
    s.ctx = ctx;
    s.group = group;
    s.vr = nullptr;
    s.cam = nullptr;
    if (src != kFocusView) {
        memcpy(ctx + kCtxSlots + kFocusView * kSlotSize, ctx + kCtxSlots + src * kSlotSize, kSlotSize);
        memcpy(ctx + kCtxCamBlock + kFocusView * kCamBlockSize, ctx + kCtxCamBlock + src * kCamBlockSize,
               kCamBlockSize);
        memcpy(ctx + kCtxQuery + kFocusView * 4, ctx + kCtxQuery + src * 4, 4);
        run.swapped = true;
    }
    uint8_t* vr = VrDevice();
    if (!vr) return;
    uint8_t* eye = vr + kVrEyeData + (kFocusView - 1) * kVrEyeSize;
    const uint8_t* srcEye = vr + kVrEyeData + (src - 1) * kVrEyeSize;
    memcpy(s.eye, eye, kVrEyeSize);
    s.vr = vr;
    uint8_t data[kVrEyeSize];
    bool proj = false;
    // Focus R ends in the right Submit texture, which then is the right eye's only image: without this frame's outer
    // R copy it must stay the plain eye view, not the focus window.
    const Copy& outerR = g_copy[CopyOuterR];
    if (g_focus.on && !g_focus.projFailed && group == FocusR && !early && (!outerR.frame || !outerR.tex)) {
        if (!g_focus.noCopyLogged) {
            g_focus.noCopyLogged = true;
            Log("Quad: focus R without the focus projection in frame %llu: no outer R copy (the right eye gets the "
                "plain eye view at the focus size)", (unsigned long long)g_frameNo);
        }
    } else if (g_focus.on && !g_focus.projFailed) {
        char why[256] = "";
        uint8_t* cam = *(uint8_t**)(ctx + kCtxSlots + kFocusView * kSlotSize);
        const focus::Eye& win = WindowEyeLocked(src - 1);
        proj = CameraReady(cam, srcEye, why, sizeof(why)) && FocusEyeData(srcEye, win, data, why, sizeof(why));
        run.x0 = win.x0;
        run.y0 = win.y0;
        if (!proj) {
            if (!g_focus.missLogged) {
                g_focus.missLogged = true;
                Log("Quad: %s without the focus projection in frame %llu (%s); off after %d misses in a row",
                    kGroupNames[group], (unsigned long long)g_frameNo, why, kProjMissLimit);
            }
            if (++g_focus.projMisses >= kProjMissLimit) {
                g_focus.projFailed = true;
                Log("Quad: focus projection OFF until the next builder call (%s: %s); %s", kGroupNames[group], why,
                    g_execPatched ? "watchdog: outer only" : "the focus groups render the eye views");
                if (g_execPatched)
                    snprintf(g_autoOuter, sizeof(g_autoOuter), "focus projection check failed (%s)",
                             kGroupNames[group]);
            }
        } else {
            g_focus.projMisses = 0;
            if (!g_focus.logged[group]) {
                g_focus.logged[group] = true;
                LogFocusFirst(group, srcEye, data, cam);
            }
            memcpy(s.camProj, cam + kCamProj, kMatSize);
            s.cam = cam;
            memcpy(cam + kCamProj, data + kEyeProj, kMatSize);
            if (cull) CullLocked(ctx, group, cam, srcEye, data, run);
        }
    }
    if (!proj) memcpy(data, srcEye, kVrEyeSize);
    if (proj || src != kFocusView) {
        memcpy(eye, data, kVrEyeSize);
        run.eyeSwapped = true;
    }
    run.projected = proj;
}

// MVR gate B8: this frame's focus window of eye i as the map from outer's clip space that FocusEyeData puts into the
// focus projection.
void MvrWindowLocked(int i) {
    if (!mvrpass::On()) return;
    const bool valid = g_focus.on && !g_focus.projFailed;
    const focus::Eye& e = WindowEyeLocked(i);
    if (!valid || e.fr == e.fl || e.fb == e.ft) {
        mvrpass::SetWindow(i, false, 1, 0, 1, 0, 0, 0);
        return;
    }
    const double sx = (e.r - e.l) / (double)(e.fr - e.fl), ox = ((e.r + e.l) - (e.fr + e.fl)) / (double)(e.fr - e.fl);
    const double sy = (e.b - e.t) / (double)(e.fb - e.ft), oy = ((e.b + e.t) - (e.fb + e.ft)) / (double)(e.fb - e.ft);
    mvrpass::SetWindow(i, true, sx, ox, sy, oy, g_focus.gameW, g_focus.gameH);
}

void RestoreLocked() {
    Swap& s = g_swap;
    if (!s.ctx) return;
    uint8_t* ctx = s.ctx;
    s.ctx = nullptr;
    s.group = -1;
    if (s.cam) memcpy(s.cam + kCamProj, s.camProj, kMatSize);
    s.cam = nullptr;
    memcpy(ctx + kCtxSlots + kFocusView * kSlotSize, s.slot, kSlotSize);
    memcpy(ctx + kCtxCamBlock + kFocusView * kCamBlockSize, s.block, kCamBlockSize);
    memcpy(ctx + kCtxQuery + kFocusView * 4, &s.query, 4);
    if (s.vr) memcpy(s.vr + kVrEyeData + (kFocusView - 1) * kVrEyeSize, s.eye, kVrEyeSize);
    s.vr = nullptr;
}

// ---------------------------------------------------------------- frame bookkeeping

void FinishFrameLocked() {
    if (!g_began && !g_ended) return;  // no eye group of the VR graph ran (menu, loading)
    ++g_st.frames;
    // Outer-only mode: only the outer groups run.
    const unsigned want = g_frameSkip ? 0x3u : 0xFu;
    if (g_frameSkip) {
        ++g_st.outerFrames;
        if (g_began & ~want) {
            ++g_st.skipLeaks;
            if (!g_skipLeakLogged) {
                g_skipLeakLogged = true;
                Log("Quad: WARNING: a focus group ran in outer-only mode in frame %llu (began 0x%X): some tasks reach Execute "
                    "outside the patched executor", (unsigned long long)g_frameNo, g_began);
            }
        }
    }
    bool full = (g_began & want) == want && (g_ended & want) == want;
    if (!full) {
        ++g_st.partial;
        // In the race every frame runs all eye groups; a partial frame means the graph is gone and other tasks
        // reuse the recorded addresses. Off until the next builder call.
        if (g_graph.ok) {
            g_graph = Graph{};
            ClearGroups();
            Log("Quad: frame %llu ran only part of the eye groups (began 0x%X, ended 0x%X): race graph gone, focus "
                "groups off until the next builder call", (unsigned long long)g_frameNo, g_began, g_ended);
        }
    } else {
        ++g_st.full;
        for (int g = 0; g < kGroups; ++g)
            if (want & (1u << g)) g_st.cpuMs[g] += g_run[g].t1 - g_run[g].t0;
    }
    if (g_run[FocusL].swapped) ++g_st.swaps;
    if (g_run[FocusL].eyeSwapped) ++g_st.eyeSwaps;
    if (g_run[FocusL].projected && g_run[FocusR].projected) {
        ++g_st.projected;
        if (g_cullOn && !g_cullFault) ++g_st.cullOnFrames;
    }
    if (full && g_run[FocusL].culled && g_run[FocusR].culled) {
        ++g_st.cullFrames;
        g_st.setEye += g_run[FocusL].cull.setEye;
        g_st.queueEye += g_run[FocusL].cull.queueEye;
        for (int i = 0; i < 2; ++i) {
            const focus_cull::Result& c = g_run[FocusL + i].cull;
            g_st.cullMs[i] += c.ms;
            g_st.setFocus[i] += c.setFocus;
            g_st.queueFocus[i] += c.queueFocus;
        }
    }
    // Expected: outer L writes into the left Submit texture, the other groups into the right one. Only frames
    // after the first Submit count (track loading runs the eye groups without Submit and without the D3D11 hook).
    // With clones (M7.3, M7.4) a group writes the clone of its Submit texture.
    bool submitting = g_lastSubmit[0] && g_lastSubmit[1];
    bool outOk = g_run[OuterL].out && clones::Original(g_run[OuterL].out) == g_lastSubmit[0];
    for (int g = OuterR; g < kGroups; ++g)
        if (want & (1u << g)) outOk &= g_run[g].out && clones::Original(g_run[g].out) == g_lastSubmit[1];
    if (full && submitting && !outOk) ++g_st.outNotSubmit;
    if (full && submitting && g_detailLeft > 0) {
        --g_detailLeft;
        Log("Quad frame %llu (all four eye groups%s):", (unsigned long long)g_frameNo,
            outOk ? ", outputs are the Submit textures as expected" : ", OUTPUTS DIFFER from the Submit textures");
        for (int g = 0; g < kGroups; ++g) {
            const Run& r = g_run[g];
            char line[256];
            const char* what = "";
            if (g >= FocusL) {
                if (r.projected)
                    what = r.swapped ? "  (view 1 fields, focus projection in the camera and the VR eye data)"
                                     : "  (focus projection in the camera and the VR eye data)";
                else if (r.swapped)
                    what = r.eyeSwapped ? "  (view 1 fields and VR eye data, NO focus projection)"
                                        : "  (view 1 fields, VR eye data NOT swapped, NO focus projection)";
                else
                    what = "  (NO focus projection)";
            }
            snprintf(line, sizeof(line), "  %-8s CBeginEye +%7.3f ms  CEndEye +%7.3f ms (%.3f ms)  output %p%s",
                     kGroupNames[g], r.t0 - g_frameT0, r.t1 - g_frameT0, r.t1 - r.t0, r.out, what);
            LogRaw(line);
        }
    }
    g_began = g_ended = 0;
    for (Run& r : g_run) r = Run{};
    for (Copy& c : g_copy) c.frame = false;
}

bool Pressed(int mod, int key, bool& wasDown) {
    bool down = key && (GetAsyncKeyState(key) & 0x8000) && (!mod || (GetAsyncKeyState(mod) & 0x8000));
    bool pressed = down && !wasDown;
    wasDown = down;
    return pressed;
}

// Debug hotkeys and the outer-only decision for the frame that starts now.
void HandleToolsLocked() {
    const Config& c = Cfg();
    if (c.overlay != g_overlayCfg) {  // the ini changed (live reload)
        g_overlayCfg = c.overlay;
        g_overlay = c.overlay;
        Log("Quad: debug overlay %s (ini)", g_overlay ? "on" : "off");
    }
    if (Pressed(c.overlayKeyMod, c.overlayKey, g_overlayKeyDown)) {
        g_overlay = !g_overlay;
        Log("Quad: debug overlay %s (hotkey)", g_overlay ? "on" : "off");
    }
    if (c.followGaze != g_followCfg) {
        g_followCfg = c.followGaze;
        g_follow = c.followGaze;
        Log("Quad: focus window %s (ini)", g_follow ? "follows the gaze (DFR)" : "fixed at the center (FFR)");
    }
    if (Pressed(c.gazeKeyMod, c.gazeKey, g_gazeKeyDown)) {
        g_follow = !g_follow;
        Log("Quad: focus window %s (hotkey)", g_follow ? "follows the gaze (DFR)" : "fixed at the center (FFR)");
    }
    if (c.clones != g_clonesCfg) {
        g_clonesCfg = c.clones;
        g_clonesOn = c.clones;
        Log("Quad: clones %s (ini)", g_clonesOn ? "on" : "off");
    }
    if (Pressed(c.clonesKeyMod, c.clonesKey, g_clonesKeyDown)) {
        g_clonesOn = !g_clonesOn;
        Log("Quad: clones %s (hotkey)", g_clonesOn ? "on" : "off");
    }
    if (c.taskProfile != g_profOn.load()) {
        g_profOn = c.taskProfile;
        Log("Quad: task CPU profile %s (ini)", c.taskProfile ? "on" : "off");
    }
    if (c.focusCull != g_cullCfg) {
        g_cullCfg = c.focusCull;
        g_cullOn = c.focusCull;
        Log("Quad: focus culling %s (ini)", g_cullOn ? "on" : "off");
    }
    if (Pressed(c.cullKeyMod, c.cullKey, g_cullKeyDown)) {
        g_cullOn = !g_cullOn;
        Log("Quad: focus culling %s (hotkey)%s", g_cullOn ? "on" : "off",
            g_cullFault ? "; STOPPED for this session after an engine fault" : g_cull.failed ? "; off after failed checks "
            "until the next track load" : "");
    }
    if (mvrpass::On() && Pressed(c.mvrKeyMod, c.mvrKey, g_mvrKeyDown)) {
        g_mvrOn = !g_mvrOn;
        Log("Quad: MVR G-buffer %s (hotkey)", g_mvrOn ? "on" : "off");
    }
    if (mvrpass::On() && c.mvrEarly != g_mvrEarlyCfg) {
        g_mvrEarlyCfg = c.mvrEarly;
        mvrpass::SetEarly(c.mvrEarly);
        Log("Quad: MVR early tasks of focus L: %s (ini)", mvrpass::EarlyName(c.mvrEarly));
    }
    if (mvrpass::On() && Pressed(c.mvrKeyMod, c.mvrEarlyKey, g_mvrEarlyKeyDown)) {
        mvrpass::SetEarly((mvrpass::GetEarly() + 1) % mvrpass::kEarlyModes);
        Log("Quad: MVR early tasks of focus L: %s (hotkey)", mvrpass::EarlyName(mvrpass::GetEarly()));
    }
    if (mvrpass::On() && c.mvrForward != g_mvrForwardCfg) {
        g_mvrForwardCfg = c.mvrForward;
        mvrpass::SetForward(c.mvrForward);
        Log("Quad: MVR live forward of both eyes %s (ini)", c.mvrForward ? "on" : "off");
    }
    if (mvrpass::On() && Pressed(c.mvrKeyMod, c.mvrForwardKey, g_mvrForwardKeyDown)) {
        mvrpass::SetForward(!mvrpass::GetForward());
        Log("Quad: MVR live forward of both eyes %s (hotkey)", mvrpass::GetForward() ? "on" : "off");
    }
    if (mvrpass::On() && c.mvrForwardDepth != g_mvrFwdDepthCfg) {
        g_mvrFwdDepthCfg = c.mvrForwardDepth;
        mvrpass::SetForwardDepth(c.mvrForwardDepth);
        Log("Quad: MVR live forward depth %s (ini)", mvrpass::ForwardDepthName(c.mvrForwardDepth));
    }
    if (mvrpass::On() && Pressed(c.mvrKeyMod, c.mvrForwardDepthKey, g_mvrFwdDepthKeyDown)) {
        mvrpass::SetForwardDepth((mvrpass::GetForwardDepth() + 1) % mvrpass::kForwardDepthModes);
        Log("Quad: MVR live forward depth %s (hotkey)", mvrpass::ForwardDepthName(mvrpass::GetForwardDepth()));
    }
    g_mapFrame = g_mapNext;
    g_mapNext = false;
    if (mvrpass::On() && Pressed(c.mvrKeyMod, c.mvrMapKey, g_mvrMapKeyDown)) {
        g_mapNext = true;
        Log("Quad: MVR task map with PS resources in the next frame (hotkey; early tasks: %s)",
            mvrpass::EarlyName(mvrpass::GetEarly()));
    }
    if (mvrpass::On() && Pressed(c.mvrKeyMod, c.mvrModeKey, g_mvrModeKeyDown)) {
        using M = mvrpass::Mode;
        const M now = mvrpass::GetMode();
        const M next = now == M::Skip ? M::SkipRaw : now == M::SkipRaw ? M::NoDepth
                       : now == M::NoDepth ? M::OuterOnly : M::Skip;
        mvrpass::SetMode(next);
        Log("Quad: MVR G-buffer mode %s (hotkey)", mvrpass::ModeName(next));
    }
    if (c.sunEffects != g_sunCfg) {
        g_sunCfg = c.sunEffects;
        g_sunOn = c.sunEffects;
        Log("Quad: sun effects from outer %s (ini)", g_sunOn ? "on" : "off");
    }
    if (Pressed(c.sunKeyMod, c.sunKey, g_sunKeyDown)) {
        g_sunOn = !g_sunOn;
        Log("Quad: sun effects from outer %s (hotkey)", g_sunOn ? "on" : "off");
    }
    if (c.taaViews != g_taaCfg) {
        g_taaCfg = c.taaViews;
        g_taaOn = c.taaViews;
        Log("Quad: own TAA state of the focus views %s (ini)", g_taaOn ? "on" : "off");
    }
    if (Pressed(c.taaViewsKeyMod, c.taaViewsKey, g_taaKeyDown)) {
        g_taaOn = !g_taaOn;
        Log("Quad: own TAA state of the focus views %s (hotkey)", g_taaOn ? "on" : "off");
    }
    if (Pressed(c.freezeKeyMod, c.freezeKey, g_freezeKeyDown)) {
        g_gazeFrozen = !g_gazeFrozen;
        Log("Quad: gaze %s (hotkey), NDC L (%.3f, %.3f) R (%.3f, %.3f)%s", g_gazeFrozen ? "FROZEN" : "live",
            g_gaze[0][0], g_gaze[0][1], g_gaze[1][0], g_gaze[1][1], g_gazeOk ? "" : ", no valid gaze");
    }
    if (Pressed(c.outerKeyMod, c.outerKey, g_outerKeyDown)) {
        if (g_autoOuter[0]) {
            Log("Quad: outer-only hotkey: back to quad, watchdog reason cleared (%s)", g_autoOuter);
            g_autoOuter[0] = 0;
            g_userOuter = false;
        } else {
            g_userOuter = !g_userOuter;
            const char* na = !g_execPatched     ? "; NOT AVAILABLE: the task executor is not patched"
                             : !g_skipSet.load() ? "; NOT AVAILABLE for this graph: focus tasks not recognized (see the "
                                                   "builder lines)"
                                                 : "";
            Log("Quad: %s (hotkey)%s", g_userOuter ? "outer only requested" : "quad requested", na);
        }
        g_noComposeRun = 0;
    }
    if (g_skipSet.load() && !GroupsAliveLocked()) {
        ClearGroups();
        Log("Quad: focus task groups gone (frame %llu); nothing skipped until the next builder call",
            (unsigned long long)g_frameNo);
    }
    const bool want = g_userOuter || g_autoOuter[0];
    const bool skip = want && g_execPatched && g_skipSet.load();
    if (skip != g_frameSkip)
        Log("Quad: frame %llu %s", (unsigned long long)g_frameNo,
            skip ? (g_autoOuter[0] ? "OUTER ONLY (watchdog)" : "OUTER ONLY (hotkey)") : "quad");
    g_frameSkip = skip;
    g_skip = skip;
}

void HandleShotLocked() {
    if (g_shotFrame) {
        g_shotFrame = false;
        shots::End(gpu::Context());
    }
    const Config& c = Cfg();
    bool down = c.shotKey && (GetAsyncKeyState(c.shotKey) & 0x8000) &&
                (!c.shotKeyMod || (GetAsyncKeyState(c.shotKeyMod) & 0x8000));
    bool pressed = down && !g_keyWasDown;
    g_keyWasDown = down;
    bool autoShot = !g_autoShotDone && c.shotAutoS > 0 && g_firstFocusMs >= 0 &&
                    NowMs() - g_firstFocusMs >= c.shotAutoS * 1000.0;
    if (!pressed && !autoShot) return;
    if (autoShot) g_autoShotDone = true;
    const char* why = autoShot ? "shot_auto_s" : "hotkey";
    // The blend band may have changed (live reload).
    for (int i = 0; i < 2; ++i) g_shotPairs[3 + i].inset = SharpInsetLocked();
    // Diagnostics (M7.4): focus's clones join the shot, grabbed at the end of focus L.
    const int dumps =
        c.shotClones ? clones::ListForShot(clones::Focus, g_dumpRes, g_dumpNames, kShotImages - kShotFixed) : 0;
    for (int i = 0; i < dumps; ++i) g_shotNames[kShotFixed + i] = g_dumpNames[i];
    g_shotCount = kShotFixed + dumps;
    g_allPairCount = (int)std::size(g_shotPairs);
    std::copy(std::begin(g_shotPairs), std::end(g_shotPairs), g_allPairs);
    if (gpu::Context() && g_graph.ok && !shots::Active()) {
        // MVR gate B7: its images after the clones' (outer L's G-buffer pass once more with MVR in this frame).
        int pairs = 0;
        g_shotCount += mvrpass::PrepareShot(g_shotNames, g_shotCount, kShotImages - g_shotCount,
                                            g_allPairs + g_allPairCount, kMvrPairs, &pairs);
        g_allPairCount += pairs;
    }
    if (!gpu::Context() || !g_graph.ok) {
        Log("Quad: shot (%s) skipped: no focus groups or no D3D11 device yet", why);
    } else if (!shots::Begin(g_shotNames, g_shotCount, g_allPairs, g_allPairCount)) {
        mvrpass::CancelShot();
        Log("Quad: shot (%s) skipped: the previous one is still being written", why);
    } else {
        g_shotFrame = true;
        Log("Quad: shot (%s) in frame %llu", why, (unsigned long long)g_frameNo + 1);
        // MVR jitter diagnosis (B9c): which contexts set which projection offset in the shot frame.
        if (c.dlaa && c.mvrForwardTest) dlaa::TraceJitter(64);
    }
}

void ReleaseCopies() {
    for (Copy& c : g_copy) {
        if (c.tex) c.tex->Release();
        c = Copy{};
    }
}

// Right after a group's CEndEye the right Submit texture holds its image; the later groups overwrite it.
void CopyLocked(int k, ID3D11Resource* res) {
    Copy& c = g_copy[k];
    ID3D11DeviceContext* imm = gpu::Context();
    if (!imm || c.failed) return;
    ID3D11Texture2D* src = nullptr;
    if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&src)) || !src) return;
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    const bool eyeSize = (d.Width == g_focus.gameW && d.Height == g_focus.gameH) ||
                         (d.Width == g_focus.outerW && d.Height == g_focus.outerH) ||
                         (d.Width == g_focus.focusTexW && d.Height == g_focus.focusTexH);
    if (!eyeSize) {
        // Not an eye texture (the desktop mirror after leaving the race in the first M4.3 run).
        if (!c.sizeLogged) {
            c.sizeLogged = true;
            Log("Quad: %s output %u x %u fmt %d is not the game's, outer's or focus's size; no copy", kCopyNames[k], d.Width,
                d.Height, (int)d.Format);
        }
        src->Release();
        return;
    }
    D3D11_TEXTURE2D_DESC have{};
    if (c.tex) c.tex->GetDesc(&have);
    if (c.tex && (have.Width != d.Width || have.Height != d.Height || have.Format != d.Format ||
                  have.MipLevels != d.MipLevels || have.ArraySize != d.ArraySize)) {
        c.tex->Release();
        c.tex = nullptr;
    }
    if (!c.tex) {
        D3D11_TEXTURE2D_DESC n = d;
        n.Usage = D3D11_USAGE_DEFAULT;
        n.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        n.CPUAccessFlags = 0;
        n.MiscFlags = 0;
        ID3D11Device* dev = nullptr;
        imm->GetDevice(&dev);
        HRESULT hr = dev ? dev->CreateTexture2D(&n, nullptr, &c.tex) : E_FAIL;
        if (dev) dev->Release();
        if (FAILED(hr) || !c.tex) {
            c.tex = nullptr;
            c.failed = true;
            Log("Quad: cannot create the %s copy %u x %u fmt %d (0x%08lX)", kCopyNames[k], d.Width, d.Height,
                (int)d.Format, (unsigned long)hr);
            src->Release();
            return;
        }
        Log("Quad: %s copy %p (%u x %u fmt %d, game texture bind flags 0x%X)", kCopyNames[k], (void*)c.tex, d.Width,
            d.Height, (int)d.Format, d.BindFlags);
    }
    imm->CopyResource(c.tex, src);
    c.frame = true;
    src->Release();
}

bool FullBounds(const vr::VRTextureBounds_t* b) {
    return !b || (b->uMin == 0.0f && b->vMin == 0.0f && b->uMax == 1.0f && b->vMax == 1.0f);
}

void NoComposeLocked(int e, const char* why) {
    if (strcmp(g_noCompose[e], why) == 0) return;
    snprintf(g_noCompose[e], sizeof(g_noCompose[e]), "%s", why);
    Log("Quad: %s eye without the composition (%s): outer only at the game's render size%s", e ? "right" : "left", why,
        e ? " if this frame's outer R copy exists" : "");
}

// The composed eye for this Submit, or null (the reason logged once). Outer L is the left Submit texture, outer R the
// copy from its CEndEye; focus L the copy from its CEndEye, focus R what its CEndEye wrote.
// The sources must be this frame's images: outer L in the left Submit texture; outer R, focus L and focus R one after
// another in the right one (the copies were taken from it at their CEndEye). Review finding 6 (M4). With clones (M7.3,
// M7.4) a group writes the clone of its Submit texture instead (outer's and focus's are different clones).
bool OutputsMatchLocked(int e, void* submitTex) {
    void* right = g_run[OuterR].out;
    void* focus = g_run[FocusR].out;
    if (!right || !focus || g_run[FocusL].out != focus || clones::Original(focus) != clones::Original(right))
        return false;
    return clones::Original(e ? right : g_run[OuterL].out) == submitTex;
}

// The size of a 2D texture.
bool TexSize(void* res, uint32_t& w, uint32_t& h) {
    ID3D11Texture2D* t = nullptr;
    if (!res || FAILED(static_cast<IUnknown*>(res)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) || !t)
        return false;
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    t->Release();
    w = d.Width;
    h = d.Height;
    return true;
}

ID3D11Texture2D* ComposeLocked(int e, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds) {
    ID3D11DeviceContext* imm = gpu::Context();
    const Copy& copy = g_copy[e ? CopyOuterR : CopyFocusL];
    const char* why = nullptr;
    if (!imm)
        why = "no D3D11 device yet";
    else if (g_began != 0xF || g_ended != 0xF)
        why = "not all four eye groups ran in this frame";
    else if (!FullBounds(bounds))
        why = "Submit with texture bounds";
    else if (!OutputsMatchLocked(e, tex->handle))
        why = "the eye groups did not write this frame's Submit textures";
    else if (!g_run[e ? FocusR : FocusL].projected)
        why = "no focus projection in this frame";
    else if (!copy.frame || !copy.tex)
        why = e ? "no outer R copy in this frame" : "no focus L copy in this frame";
    if (why) {
        NoComposeLocked(e, why);
        return nullptr;
    }
    // Left eye: outer L is what its CEndEye wrote (the left Submit texture, or its clone). Right eye: focus R is what its
    // CEndEye wrote (the right Submit texture, or its clone).
    void* src = e ? g_run[FocusR].out : g_run[OuterL].out;
    ID3D11Texture2D* game = nullptr;
    if (FAILED(static_cast<IUnknown*>(src)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&game)) || !game) {
        NoComposeLocked(e, e ? "the focus R output is not an ID3D11Texture2D" : "the outer L output is not an ID3D11Texture2D");
        return nullptr;
    }
    ID3D11Texture2D* outerTex = e ? copy.tex : game;
    ID3D11Texture2D* focusTex = e ? game : copy.tex;
    const focus::Geometry& g = g_focus.geo;
    compose::Layout l;
    l.gridW = g.gridW;
    l.gridH = g.gridH;
    // The sizes the groups rendered at: their own with clones (M7.4), else the game's.
    if (!TexSize(outerTex, l.outerW, l.outerH) || !TexSize(focusTex, l.focusTexW, l.focusTexH)) {
        game->Release();
        NoComposeLocked(e, "no size of the outer or focus texture");
        return nullptr;
    }
    l.focusW = g.focusW;
    l.focusH = g.focusH;
    // The window the eye's focus group rendered (this frame's, checked above through `projected`).
    const Run& foc = g_run[e ? FocusR : FocusL];
    l.x0 = foc.x0;
    l.y0 = foc.y0;
    const Config& c = Cfg();
    l.marginPx = (float)c.quadMarginPx;
    l.smoothen = c.smoothen;
    l.legacyEdge = c.legacyEdge;
    l.featherPx = (float)c.quadFeatherPx;
    l.overlay = g_overlay;
    if (g_gazeOk) {
        // NDC of the full eye projection (x right, y up) to grid pixels (row 0 at the top).
        l.gazeMode = g_gazeFrozen ? 2 : 1;
        l.gazeX = (g_gaze[e][0] + 1) * 0.5f * g.gridW;
        l.gazeY = (1 - g_gaze[e][1]) * 0.5f * g.gridH;
    }
    const gpu::Phase phase = e ? gpu::ComposeR : gpu::ComposeL;
    gpu::Begin(phase);
    ID3D11Texture2D* out = compose::Draw(imm, e, outerTex, focusTex, l);
    gpu::End(phase);
    game->Release();
    if (!out) NoComposeLocked(e, "composition failed, see the Compose line");
    return out;
}

// ---------------------------------------------------------------- engine detours

using Fn4 = void*(__fastcall*)(void*, void*, void*, void*);
Fn4 o_builder = nullptr, o_beginCtor = nullptr, o_endCtor = nullptr, o_bonnetCtor = nullptr, o_addTask = nullptr;
Fn4 o_beginExec = nullptr, o_endExec = nullptr;

void* __fastcall D_Builder(void* a, void* b, void* c, void* d) {
    // Not gated by the fault flag: the restore, and after a fault the graph stays empty (no focus groups).
    Guard("builder", [&] {
        std::lock_guard lock(g_m);
        ++g_builderCalls;
        RestoreLocked();
        g_graph = Graph{};
        ClearGroups();  // the old graph's tasks are freed in the builder
        g_adds.clear();
        g_focus.projFailed = false;
        g_focus.projMisses = 0;
        g_focus.missLogged = false;
        g_cull = Cull{};
        if (g_autoOuter[0]) {
            Log("Quad: new graph: watchdog outer only cleared (%s)", g_autoOuter);
            g_autoOuter[0] = 0;
        }
        g_noComposeRun = 0;
        // After a fault the new graph is built without the focus groups: outer only at the focus size.
        if (g_fault.load() && g_patched) {
            g_patched = !WritePatch(false);
            Log("Quad: builder call after an exception: %s", g_patched ? "builder patch could not be removed"
                                                                       : "no focus groups in this graph (outer only)");
        }
    });
    t_inBuilder = true;
    t_buildGroup = -1;
    void* r = o_builder(a, b, c, d);
    t_inBuilder = false;
    t_buildGroup = -1;
    Safe("builder (after)", [&] {
        std::lock_guard lock(g_m);
        Graph& gr = g_graph;
        gr.ok = !gr.unexpected && gr.begins == kGroups && gr.ends == kGroups;
        for (int g = 0; g < kGroups; ++g) gr.ok &= gr.begin[g] && gr.end[g];
        Log("Quad: builder call %d: %d CBeginEye, %d CEndEye, %d CCaptureBonnetReflection; groups (CBeginEye/CEndEye): "
            "outer L %p/%p, outer R %p/%p, focus L %p/%p, focus R %p/%p; %s",
            g_builderCalls, gr.begins, gr.ends, gr.bonnets, gr.begin[0], gr.end[0], gr.begin[1], gr.end[1], gr.begin[2],
            gr.end[2], gr.begin[3], gr.end[3],
            gr.ok ? "focus groups ready" : "UNEXPECTED graph: focus swaps off (the focus groups render the right eye)");
        BuildSkipSetLocked();
        BuildProfileLocked();
        g_adds.clear();
        g_adds.shrink_to_fit();
    });
    return r;
}

// Views 3 and 4 do not fit the render context arrays: inside the builder their tasks get view 2. The mapping does not
// depend on the fault flag, so a fault never lets view 3 or 4 reach the engine.
void* MapView(void* view) {
    int v = (int)(intptr_t)view;
    return t_inBuilder && v > kFocusView ? (void*)(intptr_t)kFocusView : view;
}

void* __fastcall D_BeginCtor(void* task, void* ctx, void* view, void* d) {
    void* r = o_beginCtor(task, ctx, MapView(view), d);
    if (t_inBuilder) {
        Safe("CBeginEye ctor", [&] {
            std::lock_guard lock(g_m);
            int i = g_graph.begins++;
            int v = (int)(intptr_t)view;
            t_buildGroup = i < kGroups ? i : -1;
            if (i < kGroups) g_graph.begin[i] = task;
            if (i >= kGroups || v != i + 1) {
                g_graph.unexpected = true;
                Log("Quad: unexpected CBeginEye(%d) as eye task %d in the builder", v, i + 1);
            }
        });
    }
    return r;
}

void* __fastcall D_EndCtor(void* task, void* ctx, void* view, void* d) {
    void* r = o_endCtor(task, ctx, view, d);
    if (t_inBuilder) {
        Safe("CEndEye ctor", [&] {
            std::lock_guard lock(g_m);
            ++g_graph.ends;
            int g = t_buildGroup;
            if (g >= 0 && !g_graph.end[g]) {
                g_graph.end[g] = task;
            } else {
                g_graph.unexpected = true;
                Log("Quad: unexpected CEndEye(%d) in the builder (group %d)", (int)(intptr_t)view, g + 1);
            }
        });
    }
    return r;
}

// Inside the builder every "add task" call is recorded: the focus groups' tasks for the outer-only switch.
void* __fastcall D_AddTask(void* group, void* task, void* c, void* d) {
    if (t_inBuilder) {
        Safe("add task", [&] {
            std::lock_guard lock(g_m);
            if (g_adds.size() < kMaxAdds) g_adds.push_back({group, task});
        });
    }
    return o_addTask(group, task, c, d);
}

void* __fastcall D_BonnetCtor(void* task, void* ctx, void* view, void* d) {
    void* r = o_bonnetCtor(task, ctx, MapView(view), d);
    if (t_inBuilder) {
        Safe("CCaptureBonnetReflection ctor", [&] {
            std::lock_guard lock(g_m);
            ++g_graph.bonnets;
        });
    }
    return r;
}

void* __fastcall D_BeginExec(void* task, void* b, void* c, void* d) {
    Safe("CBeginEye", [&] {
        std::lock_guard lock(g_m);
        int g = Find(g_graph.begin, task);
        if (g < 0) return;
        g_renderTid.store(GetCurrentThreadId(), std::memory_order_relaxed);
        if (g_swap.ctx) {  // the previous group did not end
            ++g_st.lateRestores;
            RestoreLocked();
        }
        Run& r = g_run[g];
        r = Run{};
        r.t0 = NowMs();
        g_began |= 1u << g;
        gpu::Begin((gpu::Phase)g);  // the groups and the GPU phases have the same order
        dlaa::BeginGroup(g);        // its own DLSS history per eye group
        // The eye of this group's focus work (outer R's group runs focus R's early tasks and forward pass): focus R's
        // clones are its twins (B9d).
        clones::SetFocusEye(g == OuterR || g == FocusR ? 1 : 0);
        if (!g_graph.ok) return;
        // The groups run in build order; a task without its predecessors in this frame is a stale address (the race
        // graph is gone and the menu reuses the memory).
        const unsigned before = (1u << g) - 1;
        if ((g_ended & before) != before) return;
        if (g < FocusL) MvrWindowLocked(g);
        if (g < FocusL) {
            clones::BeginGroup(gpu::Context(), clones::Outer);
            sunfx::BeginGroup(g, 0, 0, 1, 1);
            taaviews::BeginGroup(g);  // outer: only the shot-frame dump
            return;
        }
        if (g_firstFocusMs < 0) {
            g_firstFocusMs = NowMs();
            Log("Quad: first frame with focus groups (frame %llu, thread %lu)", (unsigned long long)g_frameNo,
                GetCurrentThreadId());
        }
        uint8_t* ctx = *(uint8_t**)((uint8_t*)task + kTaskCtx);
        int view = *(int*)((uint8_t*)task + kTaskView);
        if (!ctx || view != kFocusView) return;
        SwapInLocked(ctx, g, kSourceView[g], r);
        mvrpass::FocusGroup(g - FocusL, r.projected);
        clones::BeginGroup(gpu::Context(), clones::Focus);
        mvrpass::FocusGroupIn(g - FocusL);
        taaviews::BeginGroup(g);
        // The focus window as a part of outer (which covers the whole grid); without the focus projection the
        // group renders outer's view.
        const focus::Geometry& geo = g_focus.geo;
        if (r.projected && geo.gridW && geo.gridH)
            sunfx::BeginGroup(g, (float)r.x0 / geo.gridW, (float)r.y0 / geo.gridH, (float)geo.focusW / geo.gridW,
                              (float)geo.focusH / geo.gridH);
        else
            sunfx::BeginGroup(g, 0, 0, 1, 1);
    });
    return o_beginExec(task, b, c, d);
}

void* __fastcall D_EndExec(void* task, void* b, void* c, void* d) {
    sunfx::EndGroup();  // before CEndEye's copies and the desktop mirror
    dlaa::EndGroup();
    t_inEndEye = true;
    t_endRes = nullptr;
    void* r = o_endExec(task, b, c, d);
    t_inEndEye = false;
    ID3D11Resource* res = t_endRes;
    t_endRes = nullptr;
    // Not gated by the fault flag: without the restore focus R would render with the left eye's data.
    Guard("CEndEye", [&] {
        std::lock_guard lock(g_m);
        int g = Find(g_graph.end, task);
        if (g < 0) return;
        if (g_swap.ctx && g_swap.group == g) RestoreLocked();
        clones::EndGroup(gpu::Context());  // no-op without its BeginGroup
        taaviews::EndGroup();              // no-op without its BeginGroup
        if (g_fault.load(std::memory_order_relaxed)) return;
        Run& run = g_run[g];
        run.t1 = NowMs();
        g_ended |= 1u << g;
        gpu::End((gpu::Phase)g);
        run.out = res;
        if (!res) return;
        if (!g_outLogged[g]) {
            g_outLogged[g] = true;
            Log("Quad: %s writes its final image into %s", kGroupNames[g], DescribeTex(res).c_str());
        }
        if (g == OuterL) TexSize(res, g_focus.outerOutW, g_focus.outerOutH);
        if (g_shotFrame) shots::Grab(gpu::Context(), res, g);
        // Diagnostics: every clone of focus as focus L left it.
        if (g_shotFrame && g == FocusL)
            for (int i = kShotFixed; i < g_shotCount; ++i) shots::Grab(gpu::Context(), g_dumpRes[i - kShotFixed], i);
        // Only after this frame's earlier groups (a stale task address otherwise); in outer-only mode nothing is composed.
        const unsigned upTo = (2u << g) - 1;
        if (g_focus.on && g_graph.ok && !g_frameSkip && (g_began & upTo) == upTo) {
            if (g == OuterR) CopyLocked(CopyOuterR, res);
            if (g == FocusL && Cfg().quadCompose) CopyLocked(CopyFocusL, res);
        }
    });
    if (res) res->Release();
    return r;
}

// ---------------------------------------------------------------- MVR forward step 1 (B9a)

void __fastcall NoFinish(void*) {}

// A leaf task's Execute without its Finish: Execute ends with `jmp [vtable+0x38]`, so for the call the task gets a copy
// of its vtable whose slot 7 returns. The executor still runs the task later, and the proxy then skips it (Finish
// only), so the group's scheduler sees it once. `fn`: the Execute to call (a detour's original), else vtable +0x48.
constexpr int kVtCopy = 16;
bool ExecuteOnly(uint8_t* task, Fn4 fn) {
    void** vt = *(void***)task;
    void* copy[kVtCopy];
    if (!hooks::TryRead(vt, copy, sizeof(copy))) return false;
    copy[kFinishSlot] = (void*)&NoFinish;
    bool ok = true;
    *(void***)task = copy;
    __try {
        if (fn)
            fn(task, nullptr, nullptr, nullptr);
        else
            ((ExecuteFn)vt[kExecuteSlot])(task);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ok = false;
    }
    *(void***)task = vt;
    return ok;
}

// The engine's command buffer recording (Ghidra, 2026-11-07): a *BuildCommandThread task queues each recording through
// device+0x558 (a queue per worker at device+0xe1d0 + k * 0x30: u16 counters at +0x2 and +0xa, index mask at +0x24;
// k = 0 is the shared one) and releases its dependents at once (0xcde2e0, state 3), so the render thread goes on while
// workers 1..n-1 (n at device+0xf190) record. A worker takes a job by clearing its bit in device+0xe1c0 and sets it
// again when done; the command buffer (vtable 0x2111cb8) has +0x98 set while it records, and its replaying task waits
// for it (+0x60). Recordings read the engine objects' D3D11 views and sizes, so they must not run while the proxy has
// focus's clones in those objects: a recording of outer's then holds clone views (B9a run 3, mode 3 flickered).
constexpr uintptr_t kEngineDevice = 0x2684ea0;  // DAT_142684ea0: pointer to the engine device (RVA)
constexpr size_t kDevWorkers = 0xf190, kDevIdle = 0xe1c0, kDevQueues = 0xe1d0, kDevQueueStride = 0x30;
constexpr size_t kQueueA = 0x2, kQueueB = 0xa, kQueueMask = 0x24;
constexpr uintptr_t kCmdBufVtable = 0x2111cb8;
constexpr size_t kCmdBufBusy = 0x98;
constexpr size_t kCtxBufFirst = 0x7e0, kCtxBufEnd = 0x8b8;  // the render context's command buffers (0x7d96f0)
constexpr double kWorkerWaitMs = 2.0;

// True when no recording is queued or running; `why` gets the reason when the state cannot be read.
bool EngineIdle(uint8_t* ctx, const char** why) {
    uint8_t* dev = nullptr;
    uint32_t n = 0, idle = 0;
    if (!hooks::TryRead((void*)(hooks::ExeBase() + kEngineDevice), &dev, 8) || !dev ||
        !hooks::TryRead(dev + kDevWorkers, &n, 4) || n < 1 || n > 32 || !hooks::TryRead(dev + kDevIdle, &idle, 4)) {
        *why = "the engine's worker state is unreadable";
        return false;
    }
    for (uint32_t k = 0; k < n; ++k) {
        uint8_t* q = dev + kDevQueues + k * kDevQueueStride;
        uint16_t a = 0, b = 0;
        uint32_t mask = 0;
        if (!hooks::TryRead(q + kQueueA, &a, 2) || !hooks::TryRead(q + kQueueB, &b, 2) ||
            !hooks::TryRead(q + kQueueMask, &mask, 4)) {
            *why = "the engine's worker state is unreadable";
            return false;
        }
        if (((uint32_t)a - (uint32_t)b) & mask) return false;  // queued
        if (k && !(idle & (1u << k))) return false;            // worker k busy
    }
    const uintptr_t vt = hooks::ExeBase() + kCmdBufVtable;
    for (size_t off = kCtxBufFirst; off < kCtxBufEnd; off += 8) {
        uint8_t* buf = nullptr;
        uintptr_t bvt = 0;
        uint8_t busy = 0;
        if (!hooks::TryRead(ctx + off, &buf, 8) || !buf || !hooks::TryRead(buf, &bvt, 8) || bvt != vt) continue;
        if (hooks::TryRead(buf + kCmdBufBusy, &busy, 1) && busy) return false;
    }
    return true;
}

// Waits up to kWorkerWaitMs for EngineIdle. False: still busy (or unreadable, `why`).
bool WaitEngineIdle(uint8_t* ctx, const char** why, double* waitedMs) {
    LARGE_INTEGER t0{}, t{};
    QueryPerformanceCounter(&t0);
    const int64_t limit = (int64_t)(kWorkerWaitMs * g_qpcFreq / 1000.0);
    for (unsigned i = 0;; ++i) {
        const char* w = nullptr;
        const bool idle = EngineIdle(ctx, &w);
        QueryPerformanceCounter(&t);
        *waitedMs = g_qpcFreq ? (t.QuadPart - t0.QuadPart) * 1000.0 / g_qpcFreq : 0;
        if (idle) return true;
        if (w) {
            *why = w;
            return false;
        }
        if (t.QuadPart - t0.QuadPart > limit) {
            *why = "the engine's workers still recording after 2 ms";
            return false;
        }
        if (i > 64)
            SwitchToThread();
        else
            YieldProcessor();
    }
}

// Render thread, inside RunTask right before outer's CPhase3Render (eye 0 only, mvr_pass.h): focus's view is swapped
// in as its CBeginEye detour does (without the focus culling: the G-buffer pass it culls is skipped), focus's clones go
// in, the stash goes into them, then the focus group's CBeginEye, CSSAORender and CPhase2Render run without Finish;
// afterwards outer's clones state, view and CBeginEye (view number, eye target, global slot 0x13) come back.
void EarlyFocus(int eye) {
    if (!mvrpass::EarlyWanted(eye)) return;
    const int og = OuterL + eye, fg = FocusL + eye;
    uint8_t *fBegin = nullptr, *oBegin = nullptr, *ssao = nullptr, *phase2 = nullptr;
    const char* why = nullptr;
    Run run{};
    // No recording may run while focus's view and clones are in (above).
    {
        uint8_t* octx = nullptr;
        Safe("early focus (context)", [&] {
            std::lock_guard lock(g_m);
            if (g_graph.begin[og]) octx = *(uint8_t**)((uint8_t*)g_graph.begin[og] + kTaskCtx);
        });
        double waited = 0;
        if (!octx) why = "no eye groups";
        else if (!WaitEngineIdle(octx, &why, &waited)) {
            if (!why) why = "the engine's workers busy";
        }
        mvrpass::EarlyWaited(waited);
        if (why) {
            mvrpass::EarlyRefused(eye, why);
            return;
        }
    }
    Safe("early focus (swap)", [&] {
        std::lock_guard lock(g_m);
        fBegin = (uint8_t*)g_graph.begin[fg];
        oBegin = (uint8_t*)g_graph.begin[og];
        ssao = g_eyeTasks[fg].ssao;
        phase2 = g_eyeTasks[fg].phase2;
        if (!g_graph.ok || !fBegin || !oBegin) {
            why = "no eye groups";
            return;
        }
        if (!ssao || !phase2) {
            why = "the focus group has no single CSSAORender and CPhase2Render";
            return;
        }
        if (!(g_began & (1u << og)) || (g_began & (1u << fg))) {
            why = "outer's group not running or focus's already begun";
            return;
        }
        if (!g_focus.on || g_frameSkip) {
            why = "no focus window or outer only";
            return;
        }
        if (g_swap.ctx) {
            why = "a view swap is active";
            return;
        }
        if (!clones::IsReady()) {
            why = "the clones are not ready";
            return;
        }
        uint8_t* ctx = *(uint8_t**)(fBegin + kTaskCtx);
        const int view = *(int*)(fBegin + kTaskView);
        if (!ctx || view != kFocusView || ctx != *(uint8_t**)(oBegin + kTaskCtx)) {
            why = "focus's CBeginEye has another context or view";
            return;
        }
        SwapInLocked(ctx, fg, kSourceView[fg], run, false, true);
        if (!run.projected) {
            RestoreLocked();
            why = "no focus projection";
        }
    });
    if (g_fault.load(std::memory_order_relaxed)) why = "proxy fault";
    if (why) {
        mvrpass::EarlyRefused(eye, why);
        return;
    }
    LARGE_INTEGER t0{}, t1{};
    QueryPerformanceCounter(&t0);
    ID3D11DeviceContext* imm = gpu::Context();
    clones::EndGroup(imm);  // outer's (no clones of its own in the game's render size)
    clones::BeginGroup(imm, clones::Focus);
    bool ran = false, ok = true;
    if (mvrpass::GetEarly() == 3) {  // diagnosis: the swap and both CBeginEye, no task moved
        ran = true;
        ok = ExecuteOnly(fBegin, o_beginExec);
    } else if (mvrpass::EarlyBegin(eye, run.projected)) {
        ran = true;
        ok = ExecuteOnly(fBegin, o_beginExec) && ExecuteOnly(ssao, nullptr) &&
             (mvrpass::GetEarly() != 1 || ExecuteOnly(phase2, nullptr));
    }
    clones::EndGroup(imm);
    clones::BeginGroup(imm, clones::Outer);
    Guard("early focus (restore)", [&] {
        std::lock_guard lock(g_m);
        if (g_swap.ctx && g_swap.group == fg) RestoreLocked();
    });
    if (!ran) return;
    ok &= ExecuteOnly(oBegin, o_beginExec);
    QueryPerformanceCounter(&t1);
    mvrpass::EarlyEnd(eye, g_qpcFreq ? (t1.QuadPart - t0.QuadPart) * 1000.0 / g_qpcFreq : 0, ok);
    if (!ok) {
        mvrpass::SetEarly(0);
        Log("Quad: MVR early SSAO and lights: ENGINE FAULT in an early task (frame %llu); the step is off",
            (unsigned long long)g_frameNo);
    }
}

// ---------------------------------------------------------------- D3D11 detour

using OMSetRenderTargetsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                      ID3D11DepthStencilView*);
OMSetRenderTargetsFn o_omSetRt = nullptr;
constexpr int kSlotOMSetRenderTargets = 33;  // ID3D11DeviceContext vtable (d3d11.h order)

// Inside CEndEye::Execute the game first binds the eye's Submit texture and copies the view's final image into it
// (0xdb7ab0), then draws the desktop mirror (3440 x 1440 in the first M4.2 run), so only the first bind counts.
void STDMETHODCALLTYPE D_OMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                                            ID3D11DepthStencilView* dsv) {
    void* caller = _ReturnAddress();
    // Clones' bind fix: a pass with its own views of a cloned target binds the clone.
    ID3D11RenderTargetView* trtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
    ID3D11DepthStencilView* tdsv = nullptr;
    if (clones::MapTargets(c, n, rtv, dsv, trtv, &tdsv, caller)) {
        rtv = trtv;
        dsv = tdsv;
    }
    // MVR gate B8: outer's G-buffer pass binds two-slice copies.
    ID3D11RenderTargetView* mrtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
    ID3D11DepthStencilView* mdsv = nullptr;
    if (n <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && mvrpass::MapTargets(c, n, rtv, dsv, mrtv, &mdsv)) {
        rtv = mrtv;
        dsv = mdsv;
    }
    if (t_inEndEye && !t_endRes && n && rtv && rtv[0] && c == gpu::Context())
        rtv[0]->GetResource(&t_endRes);
    o_omSetRt(c, n, rtv, dsv);
    clones::OnOMSetRenderTargets(c, n, rtv, dsv, caller);
    sunfx::OnSetTargets(c, n, rtv);
    taaviews::OnSetTargets(c, n, rtv);
}

void LogStatsLocked(const Stats& s, const char* what) {
    double f = s.full ? (double)s.full : 1;
    Log("Quad stats%s: %llu in-game frames, %llu with all four eye groups, %llu partial; focus L swaps %llu (with VR eye "
        "data %llu), frames with the focus projection %llu, late restores %llu, frames with outputs other than the "
        "Submit textures %llu; Submit composed L/R %llu/%llu, outer only L/R %llu/%llu, right game texture %llu; "
        "outer-only frames %llu (focus group leaks %llu, tasks skipped so far %llu); CPU per group (CBeginEye to "
        "CEndEye, full frames): outer L %.2f ms, outer R %.2f ms, focus L %.2f ms, focus R %.2f ms",
        what, (unsigned long long)s.frames, (unsigned long long)s.full, (unsigned long long)s.partial,
        (unsigned long long)s.swaps, (unsigned long long)s.eyeSwaps, (unsigned long long)s.projected,
        (unsigned long long)s.lateRestores, (unsigned long long)s.outNotSubmit, (unsigned long long)s.composed[0],
        (unsigned long long)s.composed[1], (unsigned long long)s.outerOnly[0], (unsigned long long)s.outerOnly[1],
        (unsigned long long)s.rightAsIs, (unsigned long long)s.outerFrames, (unsigned long long)s.skipLeaks,
        (unsigned long long)g_skipped.load(), s.cpuMs[0] / f, s.cpuMs[1] / f, s.cpuMs[2] / f, s.cpuMs[3] / f);
    const uint64_t* n = s.bySource;
    const double all = (double)std::max<uint64_t>(1, n[0] + n[1] + n[2] + n[3]);
    Log("Quad focus window%s: frames on the live gaze %llu (%.1f%%), held gaze %llu (%.1f%%), fixed center %llu "
        "(%.1f%%), frozen gaze %llu (%.1f%%); gaze lost for longer than gaze_hold_s %llu times",
        what, (unsigned long long)n[SrcLive], 100 * n[SrcLive] / all, (unsigned long long)n[SrcHeld],
        100 * n[SrcHeld] / all, (unsigned long long)n[SrcFixed], 100 * n[SrcFixed] / all,
        (unsigned long long)n[SrcFrozen], 100 * n[SrcFrozen] / all, (unsigned long long)s.gazeLost);
    const double c = s.cullFrames ? (double)s.cullFrames : 1;
    Log("Quad focus culling%s: %s; frames culled %llu of %llu with it on; per focus group on the render thread (inside "
        "its CPU above) L %.3f ms, R %.3f ms; renderables eye %.0f -> focus L %.0f, R %.0f; draw queue eye %.0f -> "
        "focus L %.0f, R %.0f",
        what, g_cullFault ? "STOPPED after an engine fault" : g_cull.failed ? "OFF after failed checks" :
              g_cullOn ? "on" : "off", (unsigned long long)s.cullFrames, (unsigned long long)s.cullOnFrames,
        s.cullMs[0] / c, s.cullMs[1] / c, s.setEye / c, s.setFocus[0] / c, s.setFocus[1] / c, s.queueEye / c,
        s.queueFocus[0] / c, s.queueFocus[1] / c);
}

void AddStats(Stats& a, const Stats& b) {
    a.frames += b.frames;
    a.full += b.full;
    a.partial += b.partial;
    a.swaps += b.swaps;
    a.eyeSwaps += b.eyeSwaps;
    a.lateRestores += b.lateRestores;
    a.outNotSubmit += b.outNotSubmit;
    a.projected += b.projected;
    for (int e = 0; e < 2; ++e) {
        a.composed[e] += b.composed[e];
        a.outerOnly[e] += b.outerOnly[e];
    }
    a.rightAsIs += b.rightAsIs;
    a.outerFrames += b.outerFrames;
    a.skipLeaks += b.skipLeaks;
    for (int s = 0; s < 4; ++s) a.bySource[s] += b.bySource[s];
    a.gazeLost += b.gazeLost;
    for (int g = 0; g < kGroups; ++g) a.cpuMs[g] += b.cpuMs[g];
    a.cullFrames += b.cullFrames;
    a.cullOnFrames += b.cullOnFrames;
    a.setEye += b.setEye;
    a.queueEye += b.queueEye;
    for (int i = 0; i < 2; ++i) {
        a.cullMs[i] += b.cullMs[i];
        a.setFocus[i] += b.setFocus[i];
        a.queueFocus[i] += b.queueFocus[i];
    }
}

// Shot rectangles for the focus window of both eyes. Outer views cover the whole grid at the size outer L rendered at in
// the latest frame (outer's own with clones, else the game's), so the window in them is (x0, y0, focusW, focusH) *
// scale; the composed eye is the grid (focus in its window).
void ShotRectsLocked(const focus::Eye* eye) {
    const focus::Geometry& g = g_focus.geo;
    const uint32_t ow = g_focus.outerOutW ? g_focus.outerOutW : g_focus.gameW;
    const uint32_t oh = g_focus.outerOutH ? g_focus.outerOutH : g_focus.gameH;
    const double sx = ow / (double)g.gridW, sy = oh / (double)g.gridH;
    for (int i = 0; i < 2; ++i) {
        const focus::Eye& e = eye[i];
        shots::Pair& p = g_shotPairs[i];
        p.rx = (float)(e.x0 * sx);
        p.ry = (float)(e.y0 * sy);
        p.rw = (float)(g.focusW * sx);
        p.rh = (float)(g.focusH * sy);
        shots::Pair& f = g_shotPairs[3 + i];
        f.rx = (float)e.x0;
        f.ry = (float)e.y0;
        f.rw = (float)g.focusW;
        f.rh = (float)g.focusH;
        f.inset = SharpInsetLocked();
        shots::Pair& o = g_shotPairs[5 + i];
        o.rw = (float)g.gridW;
        o.rh = (float)g.gridH;
    }
}

// First size query of a VR session: the focus window. Only with the engine hooks and the builder patch in place, so a
// game without focus groups keeps its full size.
void SetupLocked(vr::IVRSystem* real) {
    if (g_focus.setupDone) return;
    g_focus.setupDone = true;
    static const char* const plain[2] = {"focus L vs outer L", "focus R vs outer R"};
    for (int i = 0; i < 2; ++i) g_shotPairs[i] = {g_shotPairs[i].a, g_shotPairs[i].b, plain[i]};
    for (size_t i = 3; i < std::size(g_shotPairs); ++i) {
        shots::Pair& p = g_shotPairs[i];
        p = {p.a, p.b, p.label};
    }
    if (!g_installed || !g_patched || !real) return;
    const Config& c = Cfg();
    focus::Geometry& g = g_focus.geo;
    // The window area of this session: the DFR section when it starts following the gaze, else the FFR one.
    g_focus.dfr = g_follow;
    const float secX = g_focus.dfr ? c.focusSectionX : c.fixedSectionX;
    const float secY = g_focus.dfr ? c.focusSectionY : c.fixedSectionY;
    if (!focus::Compute(real, secX, secY, 0, 0, "Quad", g)) {
        Log("Quad: no focus window (see the line above); the game keeps the SteamVR size, the focus groups render the "
            "eye views");
        return;
    }
    // Texture sizes as in mbucchia: multiplier x SteamVR size (focus: x section too). The game renders all four views
    // at its one size, so it gets the larger per axis.
    auto px = [](double v) { return (uint32_t)std::clamp(std::lround(v), 16L, 16384L); };
    g_focus.outerW = px(c.peripheralMultiplier * g.gridW);
    g_focus.outerH = px(c.peripheralMultiplier * g.gridH);
    g_focus.focusTexW = px((double)c.focusMultiplier * secX * g.gridW);
    g_focus.focusTexH = px((double)c.focusMultiplier * secY * g.gridH);
    g_focus.gameW = std::max(g_focus.outerW, g_focus.focusTexW);
    g_focus.gameH = std::max(g_focus.outerH, g_focus.focusTexH);
    g_focus.on = true;
    clones::SetSizes(g_focus.gameW, g_focus.gameH, g_focus.outerW, g_focus.outerH, g_focus.focusTexW,
                     g_focus.focusTexH);
    g_focus.engaged = true;
    for (int i = 0; i < 2; ++i) {
        float cx = 0, cy = 0;
        WindowCenter(g.eye[i], i, nullptr, cx, cy);
        focus::Place(g, g.eye[i], cx, cy);
    }
    Log("Quad: SteamVR size (1:1 grid) %u x %u; outer wants %u x %u (peripheral_multiplier %.3f); focus window %u x %u "
        "grid pixels (%s section %.3f x %.3f) wants %u x %u (focus_multiplier %.3f); size returned to the game %u x %u",
        g.gridW, g.gridH, g_focus.outerW, g_focus.outerH, c.peripheralMultiplier, g.focusW, g.focusH,
        g_focus.dfr ? "DFR" : "FFR", secX, secY, g_focus.focusTexW, g_focus.focusTexH, c.focusMultiplier, g_focus.gameW,
        g_focus.gameH);
    const bool outerOwn = g_focus.outerW != g_focus.gameW || g_focus.outerH != g_focus.gameH;
    const bool focusOwn = g_focus.focusTexW != g_focus.gameW || g_focus.focusTexH != g_focus.gameH;
    Log("Quad: smaller targets (M7.4): %s; with clones the density of 1:1 is %.0f%% x %.0f%% for outer and "
        "%.0f%% x %.0f%% for focus, without them (clones off or failed) %.0f%% x %.0f%% and %.0f%% x %.0f%%",
        outerOwn && focusOwn ? "outer and focus render into clones in their own sizes"
        : outerOwn           ? "outer renders into clones in its own size, focus at the game size"
        : focusOwn           ? "focus renders into clones in its own size, outer at the game size"
                             : "not needed, outer and focus both want the game size",
        100.0 * g_focus.outerW / g.gridW, 100.0 * g_focus.outerH / g.gridH, 100.0 * g_focus.focusTexW / g.focusW,
        100.0 * g_focus.focusTexH / g.focusH, 100.0 * g_focus.gameW / g.gridW, 100.0 * g_focus.gameH / g.gridH,
        100.0 * g_focus.gameW / g.focusW, 100.0 * g_focus.gameH / g.focusH);
    static const char* const labels[2] = {"focus L vs its window in outer L", "focus R vs its window in outer R"};
    for (int i = 0; i < 2; ++i) {
        const focus::Eye& e = g.eye[i];
        Log("Quad: %s eye: full l %.5f r %.5f t %.5f b %.5f; fixed focus (FFR) l %.5f r %.5f t %.5f b %.5f at grid "
            "pixel (%u, %u)",
            i ? "right" : "left", e.l, e.r, e.t, e.b, e.fl, e.fr, e.ft, e.fb, e.x0, e.y0);
        g_shotPairs[i].label = labels[i];
        g_window.eye[i] = e;
    }
    ShotRectsLocked(g.eye);
    g_window.frame = 0;
    g_window.src = SrcFixed;
    Log("Quad: focus window %s at start (follow_gaze=%d, gaze_hold_s=%.2f; hotkey 0x%X+0x%X switches)",
        g_follow ? "follows the gaze (DFR)" : "fixed at the center (FFR)", c.followGaze, c.gazeHoldS, c.gazeKeyMod,
        c.gazeKey);
    if (c.quadCompose) {
        // With clones the focus texture has its own size (focus_multiplier x window), else the game size.
        const bool exact = g_focus.focusTexW == g.focusW && g_focus.focusTexH == g.focusH;
        const char* how = exact ? "pixel for pixel" : "bilinearly";
        if (c.legacyEdge) {
            const int band = c.quadMarginPx + c.quadFeatherPx;
            Log("Quad: composition on: each eye goes to SteamVR as a %u x %u image, outer scaled bilinearly, focus %s "
                "with margin %d px and the pre-M7 feather %d px (%u x %u shown fully sharp)",
                g.gridW, g.gridH, how, c.quadMarginPx, c.quadFeatherPx, g.focusW > 2u * band ? g.focusW - 2 * band : 0u,
                g.focusH > 2u * band ? g.focusH - 2 * band : 0u);
            if (2u * band >= std::min(g.focusW, g.focusH))
                Log("Quad: WARNING: margin_px + feather_px = %d is half the focus size or more; the focus never shows "
                    "fully", band);
        } else {
            const double shownW = (double)g.focusW - 2.0 * c.quadMarginPx, shownH = (double)g.focusH - 2.0 * c.quadMarginPx;
            Log("Quad: composition on: each eye goes to SteamVR as a %u x %u image, outer scaled bilinearly, focus %s "
                "with margin %d px and smoothen_focus_view_edges %.3f (%.0f x %.0f shown, %.0f x %.0f fully sharp)",
                g.gridW, g.gridH, how, c.quadMarginPx, c.smoothen, std::max(0.0, shownW), std::max(0.0, shownH),
                std::max(0.0, shownW * (1 - 2 * c.smoothen)), std::max(0.0, shownH * (1 - 2 * c.smoothen)));
            if (shownW <= 0 || shownH <= 0)
                Log("Quad: WARNING: margin_px = %d is half the focus window or more; the focus never shows",
                    c.quadMarginPx);
        }
    } else {
        Log("Quad: composition off (compose=0): SteamVR gets outer at the game's render size");
    }
}

}  // namespace

void Install() {
    if (Cfg().quadMode != QuadMode::Quad || !Cfg().wrap) return;
    if (!hooks::Init()) return;
    static bool first = true;
    if (!first) {
        if (!g_installed) return;
        hooks::EnableAll();
        if (!g_patched) Guard("Install (again)", [&] { g_patched = WritePatch(true); });
        g_on = true;
        return;
    }
    first = false;
    if (!hooks::EngineAvailable()) {
        Log("Quad: exe not supported (see the Hooks lines); no engine hooks, the game renders stereo as usual");
        return;
    }
    Guard("Install", [&] {
        bool ok = true;
        ok &= hooks::Create(Anchor::BuilderOculusRift, (void*)D_Builder, (void**)&o_builder);
        ok &= hooks::Create(Anchor::BeginEyeCtor, (void*)D_BeginCtor, (void**)&o_beginCtor);
        ok &= hooks::Create(Anchor::EndEyeCtor, (void*)D_EndCtor, (void**)&o_endCtor);
        ok &= hooks::Create(Anchor::BonnetReflectionCtor, (void*)D_BonnetCtor, (void**)&o_bonnetCtor);
        ok &= hooks::Create(Anchor::BeginEyeExecute, (void*)D_BeginExec, (void**)&o_beginExec);
        ok &= hooks::Create(Anchor::EndEyeExecute, (void*)D_EndExec, (void**)&o_endExec);
        if (!ok) {
            // Without the ctor hooks the patched loop would hand views 3 and 4 to the engine.
            hooks::DisableAll();
            Log("Quad: engine hooks INCOMPLETE; no builder patch, the game renders stereo");
            return;
        }
        g_installed = true;
        Log("Quad: engine hooks installed (builder, eye task ctors, CBeginEye/CEndEye Execute)");
        // The getter returns the static app object through lea rax, [rip+disp32] at +0x20.
        const uint8_t* lea = (const uint8_t*)hooks::Address(Anchor::AppGetter) + 0x20;
        if (lea[0] == 0x48 && lea[1] == 0x8D && lea[2] == 0x05) {
            g_app = const_cast<uint8_t*>(lea) + 7 + *(const int32_t*)(lea + 3);
            Log("Quad: app object at AMS2AVX.exe+0x%llX (VR device eye data swapped with the view 2 fields)",
                (unsigned long long)((uintptr_t)g_app - hooks::ExeBase()));
        } else {
            Log("Quad: app getter does not match; eye data not swapped (focus L lighting uses the right eye)");
        }
        g_patched = WritePatch(true);
        // Clones (M7.3): the factory hooks must be in place before the engine creates its render target pool.
        if (g_patched) clones::InstallEngineHooks();
        // Sun effects: the shader creation hook must be in place before the game makes its effect shaders.
        if (g_patched) sunfx::Install();
        // MVR gate B6: the vertex shader hook must be in place before the game makes its scene shaders.
        const bool mvrVs = g_patched && Cfg().mvrVsPatch && mvrvs::Install();
        // MVR gate B9b: the pixel shader patch runs in the sun effects hook and needs NVAPI from mvr_vs.
        if (g_patched && Cfg().mvrPsPatch) {
            if (mvrVs)
                mvrps::Install();
            else
                Log("MVR PS patch: off (needs [debug] mvr_vs_patch=1)");
        }
        // TAA views: the target helper hook must be in place before the game makes its TAA (track load).
        if (g_patched) taaviews::Install();
        // Quad <-> outer only needs the focus tasks from the builder's "add task" calls and the executor patch.
        if (g_patched && hooks::Create(Anchor::TaskGroupAddTask, (void*)D_AddTask, (void**)&o_addTask))
            g_execPatched = PatchExecutor();
        if (g_execPatched) StartStallWatch();
        else if (g_patched)
            Log("Quad: no add-task hook; no quad <-> outer only switch");
        // MVR gate B8: needs the patched VS (mvr_vs) and the executor thunk (it skips the focus groups' pass).
        if (Cfg().mvrGbuffer && g_execPatched && Cfg().mvrVsPatch) {
            mvrpass::SetJitterSource(&dlaa::FrameJitter);
            mvrpass::Install();
            g_mvrEarlyCfg = Cfg().mvrEarly;  // Install took the setting
            g_mvrForwardCfg = Cfg().mvrForward;
            g_mvrFwdDepthCfg = Cfg().mvrForwardDepth;
        }
        else if (Cfg().mvrGbuffer)
            Log("MVR G-buffer: off (needs [debug] mvr_vs_patch=1 and the executor patch)");
        Log("Quad: focus culling %s (scene gather, processor and queue copy %s)", Cfg().focusCull ? "on" : "off",
            focus_cull::Available() ? "found" : "NOT FOUND: culling stays off");
    });
    g_overlay = g_overlayCfg = Cfg().overlay;
    g_follow = g_followCfg = Cfg().followGaze;
    g_clonesOn = g_clonesCfg = Cfg().clones;
    g_cullOn = g_cullCfg = Cfg().focusCull;
    g_sunOn = g_sunCfg = Cfg().sunEffects;
    g_taaOn = g_taaCfg = Cfg().taaViews;
    LARGE_INTEGER qf{};
    QueryPerformanceFrequency(&qf);
    g_qpcFreq = qf.QuadPart;
    g_profOn = Cfg().taskProfile;
    g_on = g_installed;
}

void OnFrameStart() {
    if (!g_on) return;
    g_frameStartMs = NowMs();
    // The restore is not gated by the fault flag (a swap left over from the faulting frame).
    Guard("frame", [&] {
        std::lock_guard lock(g_m);
        if (g_swap.ctx) {
            ++g_st.lateRestores;
            RestoreLocked();
        }
        if (g_fault.load(std::memory_order_relaxed)) {
            // Outer only after the fault, but never with stale group addresses (the menu reuses their memory).
            if (g_skipSet.load() && !GroupsAliveLocked()) ClearGroups();
            clones::OnFrameStart(gpu::Context(), false);  // a swap left over goes back
            sunfx::OnFrameStart(gpu::Context(), false);
            taaviews::OnFrameStart(false, false);
            return;
        }
        FinishFrameLocked();
        ++g_frameNo;
        g_frameT0 = NowMs();
        HandleToolsLocked();
        mvrpass::OnFrameStart(g_mvrOn);  // reports the comparison of the previous shot frame
        HandleShotLocked();
        clones::OnFrameStart(gpu::Context(), g_clonesOn && g_focus.on);
        sunfx::OnFrameStart(gpu::Context(), g_sunOn && g_focus.on && !g_frameSkip);
        mvrvs::OnFrameStart(gpu::Context());
        mvrps::OnFrameStart(gpu::Context());
        taaviews::OnFrameStart(g_taaOn && !g_frameSkip, g_shotFrame);
        dlaa::SetShotFrame(g_shotFrame);
    });
}

const char* SourceName(int src) {
    switch (src) {
        case SrcLive: return "live";
        case SrcHeld: return "held";
        case SrcFixed: return "fixed";
        case SrcFrozen: return "frozen";
        default: return "";
    }
}

FrameWindow SetGaze(bool ok, const float ndcL[2], const float ndcR[2], bool benchmark) {
    FrameWindow fw;
    if (!g_on) return fw;
    Safe("SetGaze", [&] {
        std::lock_guard lock(g_m);
        const double now = NowMs();
        if (!g_gazeFrozen) {
            g_gazeOk = ok;
            g_gaze[0][0] = ndcL[0];
            g_gaze[0][1] = ndcL[1];
            g_gaze[1][0] = ndcR[0];
            g_gaze[1][1] = ndcR[1];
            if (ok) {
                g_haveGood = true;
                g_goodMs = now;
                memcpy(g_good, g_gaze, sizeof(g_good));
            }
        }
        if (!g_focus.on) return;
        const Config& c = Cfg();
        // The window's center: the gaze (live, frozen, or the last valid one for gaze_hold_s) or the fixed center.
        const float(*ndc)[2] = nullptr;
        int src = SrcFixed;
        bool lost = false;
        if (g_follow && g_gazeFrozen) {
            if (g_gazeOk) {
                ndc = g_gaze;
                src = SrcFrozen;
            }
        } else if (g_follow && !benchmark) {
            if (g_gazeOk) {
                ndc = g_gaze;
                src = SrcLive;
            } else if (g_haveGood && now - g_goodMs <= c.gazeHoldS * 1000.0) {
                ndc = g_good;
                src = SrcHeld;
            } else {
                lost = true;
            }
        }
        if (lost != g_lost) {
            g_lost = lost;
            if (lost) ++g_st.gazeLost;
            if (g_lostLogs < kLostLogs && (lost || src == SrcLive)) {
                ++g_lostLogs;
                Log("Quad: frame %llu: %s%s", (unsigned long long)g_frameNo,
                    lost ? "no valid gaze for longer than gaze_hold_s, focus window at the fixed center"
                         : "valid gaze again, focus window follows it",
                    g_lostLogs == kLostLogs ? " (further losses only counted in the stats)" : "");
            }
        }
        if (benchmark != g_benchWindow) {
            g_benchWindow = benchmark;
            if (g_follow)
                Log("Quad: benchmark %s: focus window %s", benchmark ? "on" : "off",
                    !benchmark      ? "follows the gaze again"
                    : g_gazeFrozen ? "at the frozen gaze"
                                   : "at the fixed center (freeze the gaze first to measure elsewhere)");
        }
        for (int i = 0; i < 2; ++i) {
            focus::Eye& e = g_window.eye[i];
            e = g_focus.geo.eye[i];
            float cx = 0, cy = 0;
            WindowCenter(e, i, ndc ? ndc[i] : nullptr, cx, cy);
            focus::Place(g_focus.geo, e, cx, cy);
        }
        g_window.frame = g_frameNo;
        g_window.src = src;
        if (g_graph.ok) ++g_st.bySource[src];
        if (g_shotFrame) ShotRectsLocked(g_window.eye);
        fw.src = src;
        fw.x0 = g_window.eye[0].x0;
        fw.y0 = g_window.eye[0].y0;
    });
    return fw;
}

void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex) {
    if (!g_on) return;
    Safe("Submit", [&] {
        if (!tex || tex->eType != vr::API_DirectX || !tex->handle) return;
        std::lock_guard lock(g_m);
        g_lastSubmit[eye == vr::Eye_Right ? 1 : 0] = tex->handle;
        // gpu::OnSubmitTexture ran just before: a new device invalidates every object made with the old one.
        const uint32_t gen = gpu::Generation();
        if (gen != g_gen) {
            if (g_gen) Log("Quad: D3D11 device changed: copies, composition and a pending shot dropped");
            ReleaseCopies();
            compose::Release();
            shots::Cancel();
            g_shotFrame = false;
            g_gen = gen;
        }
        static bool tried = false;
        ID3D11DeviceContext* imm = gpu::Context();
        if (!imm || tried) return;
        tried = true;
        // The vtable is shared by every immediate context of d3d11.dll, so the hook survives a device change.
        void** vt = *(void***)imm;
        g_d3dHooked = hooks::CreateAt(vt[kSlotOMSetRenderTargets], (void*)D_OMSetRenderTargets, (void**)&o_omSetRt,
                                      "OMSetRenderTargets");
        Log("Quad: D3D11 hook %s (OMSetRenderTargets, finds the texture each CEndEye writes)",
            g_d3dHooked ? "installed" : "FAILED");
        if (g_d3dHooked) clones::InstallD3DHooks(imm);
        if (g_d3dHooked) mvrpass::InstallD3DHooks(imm);
    });
}

void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h) {
    if (Cfg().quadMode != QuadMode::Quad) return;
    Guard("AdjustSize", [&] {
        std::lock_guard lock(g_m);
        SetupLocked(real);
        if (!g_focus.on || !w || !h) return;
        *w = g_focus.gameW;
        *h = g_focus.gameH;
    });
}

bool Engaged() { return g_focus.engaged; }

const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds,
                                   vr::Texture_t& storage) {
    const vr::Texture_t* out = tex;
    // After a fault the per-frame state is frozen, but no swap happens any more: both game textures are eye views.
    if (!g_on || g_fault.load(std::memory_order_relaxed)) return out;
    Guard("PrepareSubmit", [&] {
        std::lock_guard lock(g_m);
        if (!g_focus.on || !g_graph.ok || !tex || tex->eType != vr::API_DirectX || !tex->handle) return;
        const int e = eye == vr::Eye_Right ? 1 : 0;
        // Outer's output when it rendered into a clone of this Submit texture (M7.4): the game texture is stale, so
        // SteamVR gets the clone (it scales a smaller texture itself).
        auto outerClone = [&](int g) -> bool {
            void* o = g_run[g].out;
            if (!o || o == tex->handle || clones::Original(o) != tex->handle) return false;
            storage = *tex;
            storage.handle = o;
            out = &storage;
            if (!g_cloneSubmitLogged) {
                g_cloneSubmitLogged = true;
                Log("Quad: %s eye goes to the compositor as outer's clone %s (no composition)", e ? "right" : "left",
                    DescribeTex(o).c_str());
            }
            return true;
        };
        // Outer only: the focus groups did not run, so both game textures (or outer's clones of them) hold outer.
        if (g_frameSkip) {
            outerClone(e ? OuterR : OuterL);
            return;
        }
        const bool fullFrame = g_began == 0xF && g_ended == 0xF;
        if (Cfg().quadCompose) {
            ID3D11Texture2D* composed = ComposeLocked(e, tex, bounds);
            // Watchdog: the focus groups ran, but the eyes keep going out without the composition.
            if (e && fullFrame) {
                g_noComposeRun = composed ? 0 : g_noComposeRun + 1;
                if (g_noComposeRun == kWatchFrames && g_execPatched && !g_autoOuter[0]) {
                    snprintf(g_autoOuter, sizeof(g_autoOuter), "%d frames without the composition (%s)",
                             kWatchFrames, g_noCompose[1][0] ? g_noCompose[1] : "see the Quad lines");
                    Log("Quad: watchdog: %s; OUTER ONLY from the next frame until the next builder call or the outer-only "
                        "hotkey", g_autoOuter);
                }
            }
            if (composed) {
                storage = *tex;
                storage.handle = composed;
                out = &storage;
                ++g_st.composed[e];
                if (g_shotFrame) shots::Grab(gpu::Context(), composed, kComposed + e);
                return;
            }
        }
        // Without the composition SteamVR gets outer alone: the left Submit texture (or outer's clone of it) holds
        // outer L, the right one ends the frame with focus R, so the right eye gets the outer R copy.
        if (!e) {
            ++g_st.outerOnly[0];
            outerClone(OuterL);
            return;
        }
        const Copy& c = g_copy[CopyOuterR];
        if (!c.frame || !c.tex || !fullFrame) {
            ++g_st.rightAsIs;
            return;
        }
        storage = *tex;
        storage.handle = c.tex;
        out = &storage;
        ++g_st.outerOnly[1];
        if (!g_rightLogged) {
            g_rightLogged = true;
            Log("Quad: right eye goes to the compositor as the outer R copy %p (game texture %p ends the frame with "
                "focus R)", (void*)c.tex, tex->handle);
        }
    });
    return out;
}

void LogStats(bool totals) {
    if (!g_installed) return;
    Safe("LogStats", [&] {
        std::lock_guard lock(g_m);
        if (!g_st.frames && !g_tot.frames) return;
        AddStats(g_tot, g_st);
        LogStatsLocked(totals ? g_tot : g_st, totals ? " (session total)" : "");
        if (!totals) LogProfileLocked(g_st);
        g_st = Stats{};
        clones::LogStats(totals);
        sunfx::LogStats(totals);
        mvrvs::LogStats(totals);
        mvrps::LogStats(totals);
        mvrpass::LogStats(totals);
        taaviews::LogStats(totals);
    });
}

std::string Describe() {
    if (Cfg().quadMode != QuadMode::Quad) return "Quad: off";
    char buf[1536] = "Quad: (error)";
    Guard("Describe", [&] {
        std::lock_guard lock(g_m);
        char focus[512] = "no focus window (the game renders at the SteamVR size)";
        const focus::Geometry& g = g_focus.geo;
        if (g_focus.on)
            snprintf(focus, sizeof(focus),
                     "game size %u x %u of %u x %u (outer wants %u x %u, focus %u x %u in a %u x %u %s window), focus "
                     "window %s (now: %s), focus projection %s, %s",
                     g_focus.gameW, g_focus.gameH, g.gridW, g.gridH, g_focus.outerW, g_focus.outerH, g_focus.focusTexW,
                     g_focus.focusTexH, g.focusW, g.focusH, g_focus.dfr ? "DFR" : "FFR",
                     g_follow ? "follows the gaze (DFR)" : "fixed (FFR)", SourceName(g_window.src),
                     g_focus.projFailed ? "OFF after a check (see log)" : "on",
                     Cfg().quadCompose ? "composition on (margin and feather in the log)"
                                       : "composition off (outer only)");
        char view[256];
        if (!g_execPatched)
            snprintf(view, sizeof(view), "no quad <-> outer only switch (executor not patched)");
        else if (g_frameSkip)
            snprintf(view, sizeof(view), "OUTER ONLY now (%s)", g_autoOuter[0] ? g_autoOuter : "hotkey");
        else
            snprintf(view, sizeof(view), "quad now");
        const char* cull = g_cullFault     ? "focus culling stopped after an engine fault"
                           : g_cull.failed ? "focus culling off after failed checks"
                           : g_cullOn      ? "focus culling on"
                                           : "focus culling off";
        snprintf(buf, sizeof(buf), "Quad: mode quad, %s, builder patch %s, focus groups %s, %s, %s, %s, %s, %s, %s, %s, %s, %s%s",
                 g_installed ? "engine hooks on" : "engine hooks off", g_patched ? "on" : "off",
                 g_graph.ok ? "running" : "not running", focus, view, cull, clones::Describe().c_str(),
                 sunfx::Describe().c_str(), taaviews::Describe().c_str(), mvrvs::Describe().c_str(),
                 mvrps::Describe().c_str(), mvrpass::Describe().c_str(), g_fault ? ", stopped after an exception" : "");
    });
    return buf;
}

const char* RunSuffix() {
    const char* s = "";
    Guard("RunSuffix", [&] {
        std::lock_guard lock(g_m);
        if (g_patched && g_graph.ok) s = g_frameSkip ? "_outeronly" : "_quad";
    });
    return s;
}

void Shutdown() {
    if (!g_installed) return;
    g_on = false;
    // The patch goes first: a builder call without the ctor hooks must not see the patched loop.
    Guard("Shutdown", [&] {
        std::lock_guard lock(g_m);
        if (g_patched) g_patched = !WritePatch(false);
    });
    // A patch still in place keeps the ctor hooks, so views 3 and 4 never reach the engine.
    if (g_patched)
        Log("Quad: builder patch still on; engine hooks stay enabled");
    else
        hooks::DisableAll();
    Guard("Shutdown (state)", [&] {
        std::lock_guard lock(g_m);
        RestoreLocked();
        clones::Shutdown();
        sunfx::Shutdown();
        taaviews::Shutdown();
        mvrpass::Shutdown();
        shots::Cancel();
        g_shotFrame = false;
        g_skip = false;
        ClearGroups();
        FreeRetiredSets();
        g_adds.clear();
        g_frameSkip = g_userOuter = false;
        g_autoOuter[0] = 0;
        g_noComposeRun = 0;
        g_skipLeakLogged = false;
        g_gazeOk = g_gazeFrozen = false;
        g_window = Window{};
        g_haveGood = g_lost = g_benchWindow = false;
        g_lostLogs = 0;
        g_gen = 0;
        g_graph = Graph{};
        g_began = g_ended = 0;
        for (Run& r : g_run) r = Run{};
        g_st = g_tot = Stats{};
        g_frameNo = 0;
        g_firstFocusMs = -1;
        g_detailLeft = 3;
        g_lastSubmit[0] = g_lastSubmit[1] = nullptr;
        for (bool& b : g_outLogged) b = false;
        g_focus = Focus{};
        g_cull = Cull{};
        ReleaseCopies();
        compose::Release();
        g_rightLogged = g_cloneSubmitLogged = false;
        for (auto& s : g_noCompose) s[0] = 0;
    });
}

}  // namespace qvfr::quad
