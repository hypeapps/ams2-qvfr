#include "quad.h"

#include <d3d11.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>

#include "common.h"
#include "compose.h"
#include "focus_geometry.h"
#include "hooks.h"
#include "shots.h"

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
constexpr size_t kCtxQuery = 0x70c;      // per view scene query (visibility list) index, stride 4
constexpr int kFocusView = 2;            // the view number the focus groups run as
constexpr size_t kAppType = 0x8;         // app singleton: type (5 or 7..9 with a VR device)
constexpr size_t kAppVr = 0xe0;          // app singleton: engine VR device (_BVROpen)
constexpr size_t kVrEyeData = 0xa0;      // VR device: per eye data (tangents, eye to head, projection), eye 0 first
constexpr size_t kVrEyeSize = 0x110;
constexpr size_t kEyeTangents = 0x0;     // eye data: 4 floats {b, -t, -l, r} computed from the projection (0xd6bf30)
constexpr size_t kEyeProj = 0x90;        // eye data: projection, row vectors, z forward (m20 = -OpenVR m02, m23 = 1)
constexpr uintptr_t kCamVtable = 0x2115d30;  // camera class vtable (RVA); all view cameras have it
constexpr size_t kCamProj = 0x110;       // camera: explicit projection (set through vtable +0x210, read by +0x138/+0x150)
constexpr size_t kCamProjDirty = 0x402;  // camera: 1 = rebuild the projection from FOV and aspect at the next update
constexpr size_t kMatSize = 0x40;

// Eye groups in build order. CBeginEye/CEndEye::Execute pick the tonemap target and the Submit texture by view - 1,
// so every group after outer L writes into the right eye's Submit texture and outer L's stays untouched.
constexpr int kGroups = 4;
enum Group { OuterL, OuterR, FocusL, FocusR };
const char* const kGroupNames[kGroups] = {"outer L", "outer R", "focus L", "focus R"};
// Shot images: the four views at their CEndEye, then the composed eyes at Submit.
constexpr int kComposed = kGroups;  // + eye
constexpr int kShotImages = kGroups + 2;
const char* const kShotNames[kShotImages] = {"outer_l", "outer_r", "focus_l", "focus_r", "composed_l", "composed_r"};
// The view whose camera, frustum and visibility list each group renders with; the focus groups replace only the
// projection (camera and VR device eye data) with the focus window of that eye.
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

// Runs f under SEH. A fault stops the focus swaps for the session (Safe skips its work from then on); the detours keep
// calling the originals, so the focus groups render the right eye like outer R.
template <class F>
void Guard(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (!g_fault.exchange(true))
            Log("ERROR: exception 0x%08lX in quad mode (%s). Focus swaps stopped for this session; the focus groups "
                "render the right eye.", code, where);
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
};

struct Stats {
    uint64_t frames = 0, full = 0, partial = 0, swaps = 0, eyeSwaps = 0, lateRestores = 0, outNotSubmit = 0;
    uint64_t projected = 0;   // frames with the focus projection in both focus groups
    uint64_t composed[2] = {};   // Submit per eye with the composed image
    uint64_t outerOnly[2] = {};  // without the composition: outer at the focus size (left: game texture, right: copy)
    uint64_t rightAsIs = 0;      // right eye: game texture (focus R), no outer R copy in this frame
    double cpuMs[kGroups] = {};
};

// Focus window for this VR session (M4.3): the game gets the focus size and the full projection (outer), the focus
// groups get the focus window of their eye.
struct Focus {
    bool setupDone = false;
    bool on = false;          // geometry ready: the game got the focus size
    bool engaged = false;     // the size was replaced in this session (the wrappers must stay)
    focus::Geometry geo;
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
char g_noCompose[2][160] = {};  // last reason logged per eye

thread_local bool t_inBuilder = false;
thread_local int t_buildGroup = -1;
thread_local bool t_inEndEye = false;
thread_local ID3D11Resource* t_endRes = nullptr;  // referenced: the RTV may be gone once CEndEye returns

// D3D11: the immediate context (no reference held; the game owns the device for the whole session).
std::atomic<ID3D11DeviceContext*> g_imm{nullptr};
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

// Before a focus group: view 2 takes the source view's camera slot, frustum block and visibility list (focus L),
// and the camera and the VR device eye data of view 2 get the focus window of the group's eye. The deferred lighting
// reads the eye tangents straight from the VR device by view - 1 (0x9ce0e8), the passes read the projection from the
// camera (+0x138, +0x150). The frustum block stays the eye's (built from FOV, a superset of the focus window).
void SwapInLocked(uint8_t* ctx, int group, int src, Run& run) {
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
    if (g_focus.on && !g_focus.projFailed && group == FocusR && (!outerR.frame || !outerR.tex)) {
        if (!g_focus.noCopyLogged) {
            g_focus.noCopyLogged = true;
            Log("Quad: focus R without the focus projection in frame %llu: no outer R copy (the right eye gets the "
                "plain eye view at the focus size)", (unsigned long long)g_frameNo);
        }
    } else if (g_focus.on && !g_focus.projFailed) {
        char why[256] = "";
        uint8_t* cam = *(uint8_t**)(ctx + kCtxSlots + kFocusView * kSlotSize);
        proj = CameraReady(cam, srcEye, why, sizeof(why)) &&
               FocusEyeData(srcEye, g_focus.geo.eye[src - 1], data, why, sizeof(why));
        if (!proj) {
            if (!g_focus.missLogged) {
                g_focus.missLogged = true;
                Log("Quad: %s without the focus projection in frame %llu (%s); off after %d misses in a row",
                    kGroupNames[group], (unsigned long long)g_frameNo, why, kProjMissLimit);
            }
            if (++g_focus.projMisses >= kProjMissLimit) {
                g_focus.projFailed = true;
                Log("Quad: focus projection OFF until the next builder call (%s: %s); the focus groups render the "
                    "eye views", kGroupNames[group], why);
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
        }
    }
    if (!proj) memcpy(data, srcEye, kVrEyeSize);
    if (proj || src != kFocusView) {
        memcpy(eye, data, kVrEyeSize);
        run.eyeSwapped = true;
    }
    run.projected = proj;
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
    bool full = g_began == 0xF && g_ended == 0xF;
    if (!full) {
        ++g_st.partial;
        // In the race every frame runs all four groups; a partial frame means the graph is gone and other tasks
        // reuse the recorded addresses. Off until the next builder call.
        if (g_graph.ok) {
            g_graph = Graph{};
            Log("Quad: frame %llu ran only part of the eye groups (began 0x%X, ended 0x%X): race graph gone, focus "
                "groups off until the next builder call", (unsigned long long)g_frameNo, g_began, g_ended);
        }
    } else {
        ++g_st.full;
        for (int g = 0; g < kGroups; ++g) g_st.cpuMs[g] += g_run[g].t1 - g_run[g].t0;
    }
    if (g_run[FocusL].swapped) ++g_st.swaps;
    if (g_run[FocusL].eyeSwapped) ++g_st.eyeSwaps;
    if (g_run[FocusL].projected && g_run[FocusR].projected) ++g_st.projected;
    // Expected: outer L writes into the left Submit texture, the other three groups into the right one. Only frames
    // after the first Submit count (track loading runs the eye groups without Submit and without the D3D11 hook).
    bool submitting = g_lastSubmit[0] && g_lastSubmit[1];
    bool outOk = g_run[OuterL].out && g_run[OuterL].out == g_lastSubmit[0];
    for (int g = OuterR; g < kGroups; ++g) outOk &= g_run[g].out && g_run[g].out == g_lastSubmit[1];
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

void HandleShotLocked() {
    if (g_shotFrame) {
        g_shotFrame = false;
        shots::End(g_imm.load());
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
    if (!g_imm.load() || !g_graph.ok) {
        Log("Quad: shot (%s) skipped: no focus groups or no D3D11 device yet", why);
    } else if (!shots::Begin(kShotNames, kShotImages, g_shotPairs, (int)std::size(g_shotPairs))) {
        Log("Quad: shot (%s) skipped: the previous one is still being written", why);
    } else {
        g_shotFrame = true;
        Log("Quad: shot (%s) in frame %llu", why, (unsigned long long)g_frameNo + 1);
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
    ID3D11DeviceContext* imm = g_imm.load();
    if (!imm || c.failed) return;
    ID3D11Texture2D* src = nullptr;
    if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&src)) || !src) return;
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    if (d.Width != g_focus.geo.focusW || d.Height != g_focus.geo.focusH) {
        // Not an eye texture (the desktop mirror after leaving the race in the first M4.3 run).
        if (!c.sizeLogged) {
            c.sizeLogged = true;
            Log("Quad: %s output %u x %u fmt %d is not the focus size; no copy", kCopyNames[k], d.Width, d.Height,
                (int)d.Format);
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
    Log("Quad: %s eye without the composition (%s): outer only at the focus size%s", e ? "right" : "left", why,
        e ? " if this frame's outer R copy exists" : "");
}

// The composed eye for this Submit, or null (the reason logged once). Outer L is the left Submit texture, outer R the
// copy from its CEndEye; focus L the copy from its CEndEye, focus R the right Submit texture.
ID3D11Texture2D* ComposeLocked(int e, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds) {
    ID3D11DeviceContext* imm = g_imm.load();
    const Copy& copy = g_copy[e ? CopyOuterR : CopyFocusL];
    const char* why = nullptr;
    if (!imm)
        why = "no D3D11 device yet";
    else if (g_began != 0xF || g_ended != 0xF)
        why = "not all four eye groups ran in this frame";
    else if (!FullBounds(bounds))
        why = "Submit with texture bounds";
    else if (!g_run[e ? FocusR : FocusL].projected)
        why = "no focus projection in this frame";
    else if (!copy.frame || !copy.tex)
        why = e ? "no outer R copy in this frame" : "no focus L copy in this frame";
    if (why) {
        NoComposeLocked(e, why);
        return nullptr;
    }
    ID3D11Texture2D* game = nullptr;
    if (FAILED(static_cast<IUnknown*>(tex->handle)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&game)) || !game) {
        NoComposeLocked(e, "the Submit texture is not an ID3D11Texture2D");
        return nullptr;
    }
    const focus::Geometry& g = g_focus.geo;
    compose::Layout l;
    l.gridW = g.gridW;
    l.gridH = g.gridH;
    l.focusW = g.focusW;
    l.focusH = g.focusH;
    l.x0 = g.eye[e].x0;
    l.y0 = g.eye[e].y0;
    l.marginPx = (float)Cfg().quadMarginPx;
    l.featherPx = (float)Cfg().quadFeatherPx;
    ID3D11Texture2D* out = compose::Draw(imm, e, e ? copy.tex : game, e ? game : copy.tex, l);
    game->Release();
    if (!out) NoComposeLocked(e, "composition failed, see the Compose line");
    return out;
}

// ---------------------------------------------------------------- engine detours

using Fn4 = void*(__fastcall*)(void*, void*, void*, void*);
Fn4 o_builder = nullptr, o_beginCtor = nullptr, o_endCtor = nullptr, o_bonnetCtor = nullptr;
Fn4 o_beginExec = nullptr, o_endExec = nullptr;

void* __fastcall D_Builder(void* a, void* b, void* c, void* d) {
    // Not gated by the fault flag: the restore, and after a fault the graph stays empty (no focus groups).
    Guard("builder", [&] {
        std::lock_guard lock(g_m);
        ++g_builderCalls;
        RestoreLocked();
        g_graph = Graph{};
        g_focus.projFailed = false;
        g_focus.projMisses = 0;
        g_focus.missLogged = false;
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
        if (g_swap.ctx) {  // the previous group did not end
            ++g_st.lateRestores;
            RestoreLocked();
        }
        Run& r = g_run[g];
        r = Run{};
        r.t0 = NowMs();
        g_began |= 1u << g;
        if (!g_graph.ok || g < FocusL) return;
        // The groups run in build order; a focus task without its predecessors in this frame is a stale address
        // (the race graph is gone and the menu reuses the memory).
        const unsigned before = (1u << g) - 1;
        if ((g_ended & before) != before) return;
        if (g_firstFocusMs < 0) {
            g_firstFocusMs = NowMs();
            Log("Quad: first frame with focus groups (frame %llu, thread %lu)", (unsigned long long)g_frameNo,
                GetCurrentThreadId());
        }
        uint8_t* ctx = *(uint8_t**)((uint8_t*)task + kTaskCtx);
        int view = *(int*)((uint8_t*)task + kTaskView);
        if (!ctx || view != kFocusView) return;
        SwapInLocked(ctx, g, kSourceView[g], r);
    });
    return o_beginExec(task, b, c, d);
}

void* __fastcall D_EndExec(void* task, void* b, void* c, void* d) {
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
        if (g_fault.load(std::memory_order_relaxed)) return;
        Run& run = g_run[g];
        run.t1 = NowMs();
        g_ended |= 1u << g;
        run.out = res;
        if (!res) return;
        if (!g_outLogged[g]) {
            g_outLogged[g] = true;
            Log("Quad: %s writes its final image into %s", kGroupNames[g], DescribeTex(res).c_str());
        }
        if (g_shotFrame) shots::Grab(g_imm.load(), res, g);
        // Only after this frame's earlier groups (a stale task address otherwise).
        const unsigned upTo = (2u << g) - 1;
        if (g_focus.on && g_graph.ok && (g_began & upTo) == upTo) {
            if (g == OuterR) CopyLocked(CopyOuterR, res);
            if (g == FocusL && Cfg().quadCompose) CopyLocked(CopyFocusL, res);
        }
    });
    if (res) res->Release();
    return r;
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
    if (t_inEndEye && !t_endRes && n && rtv && rtv[0] && c == g_imm.load(std::memory_order_relaxed))
        rtv[0]->GetResource(&t_endRes);
    o_omSetRt(c, n, rtv, dsv);
}

void LogStatsLocked(const Stats& s, const char* what) {
    double f = s.full ? (double)s.full : 1;
    Log("Quad stats%s: %llu in-game frames, %llu with all four eye groups, %llu partial; focus L swaps %llu (with VR eye "
        "data %llu), frames with the focus projection %llu, late restores %llu, frames with outputs other than the "
        "Submit textures %llu; Submit composed L/R %llu/%llu, outer only L/R %llu/%llu, right game texture %llu; CPU "
        "per group (CBeginEye to CEndEye): outer L %.2f ms, outer R %.2f ms, focus L %.2f ms, focus R %.2f ms",
        what, (unsigned long long)s.frames, (unsigned long long)s.full, (unsigned long long)s.partial,
        (unsigned long long)s.swaps, (unsigned long long)s.eyeSwaps, (unsigned long long)s.projected,
        (unsigned long long)s.lateRestores, (unsigned long long)s.outNotSubmit, (unsigned long long)s.composed[0],
        (unsigned long long)s.composed[1], (unsigned long long)s.outerOnly[0], (unsigned long long)s.outerOnly[1],
        (unsigned long long)s.rightAsIs, s.cpuMs[0] / f, s.cpuMs[1] / f, s.cpuMs[2] / f, s.cpuMs[3] / f);
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
    for (int g = 0; g < kGroups; ++g) a.cpuMs[g] += b.cpuMs[g];
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
    if (!focus::Compute(real, c.quadFocusSize, c.quadCenterX, c.quadCenterY, "Quad", g)) {
        Log("Quad: no focus window (see the line above); the game keeps the SteamVR size, the focus groups render the "
            "eye views");
        return;
    }
    g_focus.on = true;
    g_focus.engaged = true;
    Log("Quad: SteamVR size (1:1 grid) %u x %u, size returned to the game %u x %u (focus_size %.3f): outer = full FOV "
        "at %.0f%% density, focus = window at 1:1",
        g.gridW, g.gridH, g.focusW, g.focusH, c.quadFocusSize, 100.0 * g.focusW / g.gridW);
    // Outer views are focusW x focusH over the whole grid, so the focus window is (x0, y0, focusW, focusH) * scale.
    const double sx = g.focusW / (double)g.gridW, sy = g.focusH / (double)g.gridH;
    static const char* const labels[2] = {"focus L vs its window in outer L", "focus R vs its window in outer R"};
    for (int i = 0; i < 2; ++i) {
        const focus::Eye& e = g.eye[i];
        Log("Quad: %s eye: full l %.5f r %.5f t %.5f b %.5f; focus l %.5f r %.5f t %.5f b %.5f at grid pixel (%u, %u)",
            i ? "right" : "left", e.l, e.r, e.t, e.b, e.fl, e.fr, e.ft, e.fb, e.x0, e.y0);
        shots::Pair& p = g_shotPairs[i];
        p.label = labels[i];
        p.rx = (float)(e.x0 * sx);
        p.ry = (float)(e.y0 * sy);
        p.rw = (float)(g.focusW * sx);
        p.rh = (float)(g.focusH * sy);
        // The composed eye is the grid: focus 1:1 in its window, outer over the whole grid.
        shots::Pair& f = g_shotPairs[3 + i];
        f.rx = (float)e.x0;
        f.ry = (float)e.y0;
        f.rw = (float)g.focusW;
        f.rh = (float)g.focusH;
        f.inset = (float)(c.quadMarginPx + c.quadFeatherPx);
        shots::Pair& o = g_shotPairs[5 + i];
        o.rw = (float)g.gridW;
        o.rh = (float)g.gridH;
    }
    if (c.quadCompose) {
        const int band = c.quadMarginPx + c.quadFeatherPx;
        Log("Quad: composition on: each eye goes to SteamVR as a %u x %u image, outer upscaled bilinearly, focus 1:1 "
            "with margin %d px and feather %d px (%u x %u shown fully sharp)",
            g.gridW, g.gridH, c.quadMarginPx, c.quadFeatherPx, g.focusW > 2u * band ? g.focusW - 2 * band : 0u,
            g.focusH > 2u * band ? g.focusH - 2 * band : 0u);
        if (2u * band >= std::min(g.focusW, g.focusH))
            Log("Quad: WARNING: margin_px + feather_px = %d is half the focus size or more; the focus never shows fully",
                band);
    } else {
        Log("Quad: composition off (compose=0): SteamVR gets outer at the focus size");
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
    });
    g_on = g_installed;
}

void OnFrameStart() {
    if (!g_on) return;
    // The restore is not gated by the fault flag (a swap left over from the faulting frame).
    Guard("frame", [&] {
        std::lock_guard lock(g_m);
        if (g_swap.ctx) {
            ++g_st.lateRestores;
            RestoreLocked();
        }
        if (g_fault.load(std::memory_order_relaxed)) return;
        FinishFrameLocked();
        ++g_frameNo;
        g_frameT0 = NowMs();
        HandleShotLocked();
    });
}

void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex) {
    if (!g_on) return;
    Safe("Submit", [&] {
        if (!tex || tex->eType != vr::API_DirectX || !tex->handle) return;
        {
            std::lock_guard lock(g_m);
            g_lastSubmit[eye == vr::Eye_Right ? 1 : 0] = tex->handle;
        }
        if (g_imm.load()) return;
        ID3D11Texture2D* t = nullptr;
        if (FAILED(static_cast<IUnknown*>(tex->handle)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) || !t)
            return;
        ID3D11Device* dev = nullptr;
        t->GetDevice(&dev);
        t->Release();
        if (!dev) return;
        ID3D11DeviceContext* imm = nullptr;
        dev->GetImmediateContext(&imm);
        dev->Release();
        if (!imm) return;
        if (!g_d3dHooked) {
            void** vt = *(void***)imm;
            g_d3dHooked = hooks::CreateAt(vt[kSlotOMSetRenderTargets], (void*)D_OMSetRenderTargets, (void**)&o_omSetRt,
                                          "OMSetRenderTargets");
            Log("Quad: D3D11 hook %s (OMSetRenderTargets, finds the texture each CEndEye writes)",
                g_d3dHooked ? "installed" : "FAILED");
        }
        g_imm = imm;  // no reference kept
        imm->Release();
    });
}

void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h) {
    if (Cfg().quadMode != QuadMode::Quad) return;
    Guard("AdjustSize", [&] {
        std::lock_guard lock(g_m);
        SetupLocked(real);
        if (!g_focus.on || !w || !h) return;
        *w = g_focus.geo.focusW;
        *h = g_focus.geo.focusH;
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
        if (Cfg().quadCompose) {
            if (ID3D11Texture2D* composed = ComposeLocked(e, tex, bounds)) {
                storage = *tex;
                storage.handle = composed;
                out = &storage;
                ++g_st.composed[e];
                if (g_shotFrame) shots::Grab(g_imm.load(), composed, kComposed + e);
                return;
            }
        }
        // Without the composition SteamVR gets outer at the focus size: the left Submit texture holds outer L, the
        // right one ends the frame with focus R, so the right eye gets the outer R copy.
        if (!e) {
            ++g_st.outerOnly[0];
            return;
        }
        const Copy& c = g_copy[CopyOuterR];
        if (!c.frame || !c.tex || g_began != 0xF || g_ended != 0xF) {
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
        g_st = Stats{};
    });
}

std::string Describe() {
    if (Cfg().quadMode != QuadMode::Quad) return "Quad: off";
    char buf[512] = "Quad: (error)";
    Guard("Describe", [&] {
        std::lock_guard lock(g_m);
        char focus[256] = "no focus window (the game renders at the SteamVR size)";
        const focus::Geometry& g = g_focus.geo;
        if (g_focus.on)
            snprintf(focus, sizeof(focus),
                     "game size %u x %u of %u x %u (focus_size %.3f), focus projection %s, %s",
                     g.focusW, g.focusH, g.gridW, g.gridH, Cfg().quadFocusSize,
                     g_focus.projFailed ? "OFF after a check (see log)" : "on",
                     Cfg().quadCompose ? "composition on (margin and feather in the log)"
                                       : "composition off (outer only)");
        snprintf(buf, sizeof(buf), "Quad: mode quad, %s, builder patch %s, focus groups %s, %s%s",
                 g_installed ? "engine hooks on" : "engine hooks off", g_patched ? "on" : "off",
                 g_graph.ok ? "running" : "not running", focus, g_fault ? ", stopped after an exception" : "");
    });
    return buf;
}

const char* RunSuffix() {
    const char* s = "";
    Guard("RunSuffix", [&] {
        std::lock_guard lock(g_m);
        if (g_patched && g_graph.ok) s = "_quad";
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
        shots::Cancel();
        g_shotFrame = false;
        g_imm = nullptr;
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
        ReleaseCopies();
        compose::Release();
        g_rightLogged = false;
        for (auto& s : g_noCompose) s[0] = 0;
    });
}

}  // namespace qvfr::quad
