#include "sr.h"

#include <d3d11.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include "clones.h"
#include "common.h"
#include "gpu.h"
#include "hook_prof.h"
#include "hooks.h"
#include "shots.h"

namespace qvfr::sr {

namespace {

using hooks::Anchor;

std::atomic<bool> g_on{false};
std::atomic<bool> g_fault{false};
bool g_installed = false;
bool g_d3dHooked = false;
std::mutex g_m;
std::vector<void*> g_begin, g_end;  // the VR graph's CBeginEye and CEndEye tasks (last build)
thread_local bool t_inBuilder = false;
uint32_t g_gameW = 0, g_gameH = 0;
uint64_t g_groups = 0, g_groupsTot = 0, g_other = 0;
// Render thread: the VR eye group being executed (its index among the graph's CBeginEye tasks), -1 outside one.
int g_eye = -1;

// Shot (sr.h): images by index; a frame's grabs are written at the next frame start.
constexpr int kShotImages = 6;
const char* const kShotNames[kShotImages] = {"sr_group1_dlss_in", "sr_group1_dlss_out", "sr_group2_dlss_in",
                                             "sr_group2_dlss_out", "sr_submit_l", "sr_submit_r"};
bool g_shotFrame = false;
bool g_shotKeyDown = false;
bool g_shotFault = false;

// Like Guard, but a fault only turns the shots off.
template <class F>
void ShotGuard(const char* where, F&& f) {
    if (g_shotFault) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g_shotFault = true;
        g_shotFrame = false;
        Log("ERROR: exception 0x%08lX in the DLSS SR shot (%s); no more shots in this session", code, where);
    }
}

void HandleShot() {
    ID3D11DeviceContext* c = gpu::Context();
    if (g_shotFrame) {
        g_shotFrame = false;
        shots::End(c);
    }
    const Config& cfg = Cfg();
    const bool down = cfg.shotKey && (GetAsyncKeyState(cfg.shotKey) & 0x8000) &&
                      (!cfg.shotKeyMod || (GetAsyncKeyState(cfg.shotKeyMod) & 0x8000));
    const bool pressed = down && !g_shotKeyDown;
    g_shotKeyDown = down;
    if (!pressed) return;
    if (!c) {
        Log("DLSS SR: shot (hotkey) skipped: no D3D11 device yet");
    } else if (!shots::Begin(kShotNames, kShotImages, nullptr, 0)) {
        Log("DLSS SR: shot (hotkey) skipped: the previous one is still being written");
    } else {
        g_shotFrame = true;
        Log("DLSS SR: shot (hotkey): DLSS input and output of each eye group and the Submit textures in this frame");
    }
}

template <class F>
void Guard(const char* where, F&& f) {
    if (g_fault.load(std::memory_order_relaxed)) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g_fault = true;
        g_on = false;
        Log("ERROR: exception 0x%08lX in DLSS SR (%s); the game renders at the eye size from now on", code, where);
    }
}

// ---------------------------------------------------------------- engine detours

using Fn4 = void*(__fastcall*)(void*, void*, void*, void*);
Fn4 o_builder = nullptr, o_beginCtor = nullptr, o_endCtor = nullptr, o_beginExec = nullptr, o_endExec = nullptr;

void* __fastcall D_Builder(void* a, void* b, void* c, void* d) {
    Guard("builder", [&] {
        std::lock_guard lock(g_m);
        g_begin.clear();
        g_end.clear();
    });
    t_inBuilder = true;
    void* r = o_builder(a, b, c, d);
    t_inBuilder = false;
    Guard("builder (after)", [&] {
        std::lock_guard lock(g_m);
        Log("DLSS SR: VR task graph built: %zu CBeginEye, %zu CEndEye tasks", g_begin.size(), g_end.size());
    });
    return r;
}

void* __fastcall D_BeginCtor(void* a, void* b, void* c, void* d) {
    void* r = o_beginCtor(a, b, c, d);
    if (t_inBuilder) Guard("CBeginEye ctor", [&] {
        std::lock_guard lock(g_m);
        if (g_begin.size() < 8) g_begin.push_back(a);
    });
    return r;
}

void* __fastcall D_EndCtor(void* a, void* b, void* c, void* d) {
    void* r = o_endCtor(a, b, c, d);
    if (t_inBuilder) Guard("CEndEye ctor", [&] {
        std::lock_guard lock(g_m);
        if (g_end.size() < 8) g_end.push_back(a);
    });
    return r;
}

// The clones go in before CBeginEye::Execute (it binds the view's targets) and out after CEndEye::Execute, as in quad
// mode; at the TAA draw dlaa.cpp ends the scene phase earlier (clones::SceneEnd).
void* __fastcall D_BeginExec(void* task, void* b, void* c, void* d) {
    Guard("CBeginEye", [&] {
        bool vr = false;
        {
            std::lock_guard lock(g_m);
            const auto it = std::find(g_begin.begin(), g_begin.end(), task);
            vr = it != g_begin.end();
            g_eye = vr ? (int)(it - g_begin.begin()) : -1;
            if (vr)
                ++g_groups;
            else
                ++g_other;
        }
        if (vr && g_on.load(std::memory_order_relaxed)) clones::BeginGroup(gpu::Context(), clones::Outer);
    });
    return o_beginExec(task, b, c, d);
}

void* __fastcall D_EndExec(void* task, void* b, void* c, void* d) {
    void* r = o_endExec(task, b, c, d);
    // Not gated by the fault flag: a swap left in would leave the engine with the clones.
    clones::EndGroup(gpu::Context());  // guards itself; no-op without its BeginGroup
    g_eye = -1;
    return r;
}

// ---------------------------------------------------------------- D3D11

using OMSetRenderTargetsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                      ID3D11DepthStencilView*);
OMSetRenderTargetsFn o_omSetRt = nullptr;
constexpr int kSlotOMSetRenderTargets = 33;  // ID3D11DeviceContext vtable (d3d11.h order)

// The clones' bind fix and learning need every render target bind of the immediate context (quad.cpp does the same).
void STDMETHODCALLTYPE D_OMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                                            ID3D11DepthStencilView* dsv) {
    hookprof::Scope prof(hookprof::KTargets);
    void* caller = _ReturnAddress();
    ID3D11RenderTargetView* trtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
    ID3D11DepthStencilView* tdsv = nullptr;
    if (clones::MapTargets(c, n, rtv, dsv, trtv, &tdsv, caller)) {
        rtv = trtv;
        dsv = tdsv;
    }
    {
        hookprof::Orig o;
        o_omSetRt(c, n, rtv, dsv);
    }
    clones::OnOMSetRenderTargets(c, n, rtv, dsv, caller);
}

void EnsureD3DHooks() {
    if (g_d3dHooked) return;
    ID3D11DeviceContext* imm = gpu::Context();
    if (!imm) return;
    g_d3dHooked = true;
    clones::InstallD3DHooks(imm);
    void** vt = *(void***)imm;
    const bool ok = hooks::CreateAt(vt[kSlotOMSetRenderTargets], (void*)D_OMSetRenderTargets, (void**)&o_omSetRt,
                                    "OMSetRenderTargets");
    Log("DLSS SR: D3D11 OMSetRenderTargets hook %s on immediate context %p", ok ? "installed" : "FAILED", (void*)imm);
    if (!ok) g_on = false;
}

}  // namespace

void Install() {
    const Config& c = Cfg();
    if (!c.dlaaSr || c.quadMode != QuadMode::Off) return;
    if (!hooks::Init()) return;
    static bool first = true;
    if (!first) {
        if (g_installed && !g_fault) {
            hooks::EnableAll();
            g_on = true;
        }
        return;
    }
    first = false;
#ifndef QVFR_HAVE_DLSS
    Log("DLSS SR: this proxy was built without the NVIDIA DLSS SDK; the game renders as usual");
    return;
#else
    if (!hooks::EngineAvailable()) {
        Log("DLSS SR: exe not supported (see the Hooks lines); the game renders as usual");
        return;
    }
    Guard("Install", [&] {
        bool ok = hooks::Create(Anchor::BuilderOculusRift, (void*)D_Builder, (void**)&o_builder);
        ok &= hooks::Create(Anchor::BeginEyeCtor, (void*)D_BeginCtor, (void**)&o_beginCtor);
        ok &= hooks::Create(Anchor::EndEyeCtor, (void*)D_EndCtor, (void**)&o_endCtor);
        ok &= hooks::Create(Anchor::BeginEyeExecute, (void*)D_BeginExec, (void**)&o_beginExec);
        ok &= hooks::Create(Anchor::EndEyeExecute, (void*)D_EndExec, (void**)&o_endExec);
        // The factory hooks must be in place before the engine creates its render target pool.
        ok = ok && clones::InstallEngineHooks();
        if (!ok) {
            Log("DLSS SR: engine hooks INCOMPLETE; the game renders at the eye size (DLAA)");
            return;
        }
        clones::SetStereoSr(true);
        g_installed = true;
        g_on = true;
        Log("DLSS SR: engine hooks installed (builder, eye task ctors, CBeginEye/CEndEye Execute, clone factory); "
            "render scale %.3f",
            c.dlaaRenderScale);
    });
#endif
}

bool On() { return g_on.load(std::memory_order_relaxed); }

void NoteGameSize(uint32_t w, uint32_t h) {
    if (!g_installed || !w || !h) return;
    Guard("size", [&] {
        if (w == g_gameW && h == g_gameH) return;
        g_gameW = w;
        g_gameH = h;
        const float s = Cfg().dlaaRenderScale;
        const uint32_t rw = std::max(16u, (uint32_t)(w * s + 0.5f)), rh = std::max(16u, (uint32_t)(h * s + 0.5f));
        Log("DLSS SR: eye size %u x %u, scene at %u x %u (render scale %.3f)", w, h, rw, rh, s);
        clones::SetSizes(w, h, rw, rh, w, h);
    });
}

void OnFrameStart(bool wanted) {
    if (!g_installed) return;
    Guard("frame", [&] {
        EnsureD3DHooks();
        clones::OnFrameStart(gpu::Context(), wanted && g_on.load(std::memory_order_relaxed));
    });
    ShotGuard("frame", [] { HandleShot(); });
}

void ShotDlss(ID3D11DeviceContext* c, ID3D11Resource* in, ID3D11Resource* out) {
    if (!g_shotFrame || g_eye < 0 || g_eye > 1) return;
    ShotGuard("DLSS", [&] {
        shots::Grab(c, in, 2 * g_eye);
        shots::Grab(c, out, 2 * g_eye + 1);
    });
}

void OnSubmit(int eye, void* d3dTexture) {
    if (!g_shotFrame || !d3dTexture || eye < 0 || eye > 1) return;
    ShotGuard("Submit", [&] { shots::Grab(gpu::Context(), static_cast<ID3D11Resource*>(d3dTexture), 4 + eye); });
}

void Shutdown() {
    if (!g_installed) return;
    if (g_shotFrame) {
        g_shotFrame = false;
        shots::Cancel();
    }
    g_on = false;
    Guard("Shutdown", [&] {
        std::lock_guard lock(g_m);
        Log("DLSS SR: %llu VR eye groups, %llu other CBeginEye (menus)", (unsigned long long)g_groups,
            (unsigned long long)g_other);
        g_groups = g_other = 0;
    });
    clones::Shutdown();
}

}  // namespace qvfr::sr
