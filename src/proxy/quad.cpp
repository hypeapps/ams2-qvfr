#include "quad.h"

#include <d3d11.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>

#include "common.h"
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

// Eye groups in build order. CBeginEye/CEndEye::Execute pick the tonemap target and the Submit texture by view - 1,
// so every group after outer L writes into the right eye's Submit texture and outer L's stays untouched.
constexpr int kGroups = 4;
enum Group { OuterL, OuterR, FocusL, FocusR };
const char* const kGroupNames[kGroups] = {"outer L", "outer R", "focus L", "focus R"};
const char* const kShotNames[kGroups] = {"outer_l", "outer_r", "focus_l", "focus_r"};
// The view whose camera data each group renders with (M4.2: the eye cameras).
constexpr int kSourceView[kGroups] = {1, 2, 1, 2};
const shots::Pair kShotPairs[] = {
    {FocusL, OuterL, "focus L vs outer L"},
    {FocusR, OuterR, "focus R vs outer R"},
    {OuterL, OuterR, "outer L vs outer R (reference: parallax)"},
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
};

struct Run {
    double t0 = 0, t1 = 0;
    void* out = nullptr;  // texture the group's CEndEye wrote (only compared)
    bool swapped = false;
};

struct Stats {
    uint64_t frames = 0, full = 0, partial = 0, swaps = 0, lateRestores = 0, outNotSubmit = 0;
    double cpuMs[kGroups] = {};
};

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

thread_local bool t_inBuilder = false;
thread_local int t_buildGroup = -1;
thread_local bool t_inEndEye = false;
thread_local ID3D11RenderTargetView* t_endRtv = nullptr;

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

bool WritePatch(bool apply) {
    for (const PatchByte& p : kPatch) {
        uint8_t* a = (uint8_t*)hooks::Address(p.anchor);
        uint8_t want = apply ? p.from : p.to;
        if (!a || a[p.off] != want) {
            Log("Quad: builder patch not %s: %s+%zu is 0x%02X, expected 0x%02X", apply ? "applied" : "removed",
                hooks::Name(p.anchor), p.off, a ? a[p.off] : 0, want);
            return false;
        }
    }
    for (const PatchByte& p : kPatch) {
        uint8_t* a = (uint8_t*)hooks::Address(p.anchor) + p.off;
        DWORD old = 0;
        if (!VirtualProtect(a, 1, PAGE_EXECUTE_READWRITE, &old)) {
            Log("Quad: VirtualProtect failed at %s (error %lu)", hooks::Name(p.anchor), GetLastError());
            return false;
        }
        *a = apply ? p.to : p.from;
        VirtualProtect(a, 1, old, &old);
        FlushInstructionCache(GetCurrentProcess(), a, 1);
    }
    Log("Quad: builder patch %s (4 bytes: eye loop for views 2..4, group array at [rsp+0x60])",
        apply ? "applied" : "removed");
    return true;
}

// ---------------------------------------------------------------- view 2 swap

void SwapInLocked(uint8_t* ctx, int group, int src) {
    Swap& s = g_swap;
    memcpy(s.slot, ctx + kCtxSlots + kFocusView * kSlotSize, kSlotSize);
    memcpy(s.block, ctx + kCtxCamBlock + kFocusView * kCamBlockSize, kCamBlockSize);
    memcpy(&s.query, ctx + kCtxQuery + kFocusView * 4, 4);
    memcpy(ctx + kCtxSlots + kFocusView * kSlotSize, ctx + kCtxSlots + src * kSlotSize, kSlotSize);
    memcpy(ctx + kCtxCamBlock + kFocusView * kCamBlockSize, ctx + kCtxCamBlock + src * kCamBlockSize, kCamBlockSize);
    memcpy(ctx + kCtxQuery + kFocusView * 4, ctx + kCtxQuery + src * 4, 4);
    s.ctx = ctx;
    s.group = group;
}

void RestoreLocked() {
    Swap& s = g_swap;
    if (!s.ctx) return;
    uint8_t* ctx = s.ctx;
    s.ctx = nullptr;
    s.group = -1;
    memcpy(ctx + kCtxSlots + kFocusView * kSlotSize, s.slot, kSlotSize);
    memcpy(ctx + kCtxCamBlock + kFocusView * kCamBlockSize, s.block, kCamBlockSize);
    memcpy(ctx + kCtxQuery + kFocusView * 4, &s.query, 4);
}

// ---------------------------------------------------------------- frame bookkeeping

void FinishFrameLocked() {
    if (!g_began) return;  // no eye group of the VR graph ran (menu, loading)
    ++g_st.frames;
    bool full = g_began == 0xF && g_ended == 0xF;
    if (!full) {
        ++g_st.partial;
    } else {
        ++g_st.full;
        for (int g = 0; g < kGroups; ++g) g_st.cpuMs[g] += g_run[g].t1 - g_run[g].t0;
    }
    if (g_run[FocusL].swapped) ++g_st.swaps;
    // Expected: outer L writes into the left Submit texture, the other three groups into the right one.
    bool outOk = g_run[OuterL].out == g_lastSubmit[0];
    for (int g = OuterR; g < kGroups; ++g) outOk &= g_run[g].out == g_lastSubmit[1];
    if (full && !outOk) ++g_st.outNotSubmit;
    if (full && g_detailLeft > 0) {
        --g_detailLeft;
        Log("Quad frame %llu (all four eye groups%s):", (unsigned long long)g_frameNo,
            outOk ? ", outputs are the Submit textures as expected" : ", OUTPUTS DIFFER from the Submit textures");
        for (int g = 0; g < kGroups; ++g) {
            const Run& r = g_run[g];
            char line[256];
            snprintf(line, sizeof(line), "  %-8s CBeginEye +%7.3f ms  CEndEye +%7.3f ms (%.3f ms)  output %p%s", kGroupNames[g],
                     r.t0 - g_frameT0, r.t1 - g_frameT0, r.t1 - r.t0, r.out,
                     r.swapped ? "  (view 2 fields swapped for view 1)" : "");
            LogRaw(line);
        }
    }
    g_began = g_ended = 0;
    for (Run& r : g_run) r = Run{};
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
    } else if (!shots::Begin(kShotNames, kGroups, kShotPairs, (int)std::size(kShotPairs))) {
        Log("Quad: shot (%s) skipped: the previous one is still being written", why);
    } else {
        g_shotFrame = true;
        Log("Quad: shot (%s) in frame %llu", why, (unsigned long long)g_frameNo + 1);
    }
}

// ---------------------------------------------------------------- engine detours

using Fn4 = void*(__fastcall*)(void*, void*, void*, void*);
Fn4 o_builder = nullptr, o_beginCtor = nullptr, o_endCtor = nullptr, o_bonnetCtor = nullptr;
Fn4 o_beginExec = nullptr, o_endExec = nullptr;

void* __fastcall D_Builder(void* a, void* b, void* c, void* d) {
    Safe("builder", [&] {
        std::lock_guard lock(g_m);
        ++g_builderCalls;
        RestoreLocked();
        g_graph = Graph{};
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
        if (g_firstFocusMs < 0) {
            g_firstFocusMs = NowMs();
            Log("Quad: first frame with focus groups (frame %llu, thread %lu)", (unsigned long long)g_frameNo,
                GetCurrentThreadId());
        }
        int src = kSourceView[g];
        uint8_t* ctx = *(uint8_t**)((uint8_t*)task + kTaskCtx);
        int view = *(int*)((uint8_t*)task + kTaskView);
        if (src == kFocusView || !ctx || view != kFocusView) return;
        SwapInLocked(ctx, g, src);
        r.swapped = true;
    });
    return o_beginExec(task, b, c, d);
}

void* __fastcall D_EndExec(void* task, void* b, void* c, void* d) {
    t_inEndEye = true;
    t_endRtv = nullptr;
    void* r = o_endExec(task, b, c, d);
    t_inEndEye = false;
    ID3D11RenderTargetView* rtv = t_endRtv;
    t_endRtv = nullptr;
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
        ID3D11Resource* res = nullptr;
        if (rtv) rtv->GetResource(&res);
        run.out = res;
        if (!res) return;
        if (!g_outLogged[g]) {
            g_outLogged[g] = true;
            Log("Quad: %s writes its final image into %s", kGroupNames[g], DescribeTex(res).c_str());
        }
        if (g_shotFrame) shots::Grab(g_imm.load(), res, g);
        res->Release();
    });
    return r;
}

// ---------------------------------------------------------------- D3D11 detour

using OMSetRenderTargetsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                      ID3D11DepthStencilView*);
OMSetRenderTargetsFn o_omSetRt = nullptr;
constexpr int kSlotOMSetRenderTargets = 33;  // ID3D11DeviceContext vtable (d3d11.h order)

// Inside CEndEye::Execute the game binds the eye's Submit texture and copies the view's final image into it.
void STDMETHODCALLTYPE D_OMSetRenderTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv,
                                            ID3D11DepthStencilView* dsv) {
    if (t_inEndEye && n && rtv && rtv[0] && c == g_imm.load(std::memory_order_relaxed)) t_endRtv = rtv[0];
    o_omSetRt(c, n, rtv, dsv);
}

void LogStatsLocked(const Stats& s, const char* what) {
    double f = s.full ? (double)s.full : 1;
    Log("Quad stats%s: %llu in-game frames, %llu with all four eye groups, %llu partial; focus L swaps %llu, late restores "
        "%llu, frames with outputs other than the Submit textures %llu; CPU per group (CBeginEye to CEndEye): outer L "
        "%.2f ms, outer R %.2f ms, focus L %.2f ms, focus R %.2f ms",
        what, (unsigned long long)s.frames, (unsigned long long)s.full, (unsigned long long)s.partial,
        (unsigned long long)s.swaps, (unsigned long long)s.lateRestores, (unsigned long long)s.outNotSubmit,
        s.cpuMs[0] / f, s.cpuMs[1] / f, s.cpuMs[2] / f, s.cpuMs[3] / f);
}

void AddStats(Stats& a, const Stats& b) {
    a.frames += b.frames;
    a.full += b.full;
    a.partial += b.partial;
    a.swaps += b.swaps;
    a.lateRestores += b.lateRestores;
    a.outNotSubmit += b.outNotSubmit;
    for (int g = 0; g < kGroups; ++g) a.cpuMs[g] += b.cpuMs[g];
}

}  // namespace

void Install() {
    if (Cfg().quadMode != QuadMode::Quad || !Cfg().wrap) return;
    if (!hooks::Init()) return;
    static bool first = true;
    if (!first) {
        if (!g_installed) return;
        hooks::EnableAll();
        Guard("Install (again)", [&] { g_patched = WritePatch(true); });
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
        g_patched = WritePatch(true);
    });
    g_on = g_installed;
}

void OnFrameStart() {
    if (!g_on) return;
    Safe("frame", [&] {
        std::lock_guard lock(g_m);
        if (g_swap.ctx) {
            ++g_st.lateRestores;
            RestoreLocked();
        }
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
    char buf[256] = "Quad: (error)";
    Guard("Describe", [&] {
        std::lock_guard lock(g_m);
        snprintf(buf, sizeof(buf), "Quad: mode quad, %s, builder patch %s, focus groups %s (M4.2: eye cameras)%s",
                 g_installed ? "engine hooks on" : "engine hooks off", g_patched ? "on" : "off",
                 g_graph.ok ? "running" : "not running", g_fault ? ", stopped after an exception" : "");
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
    });
}

}  // namespace qvfr::quad
