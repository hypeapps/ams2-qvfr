#include "probe.h"

#include <d3d11_4.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "common.h"
#include "hooks.h"

namespace qvfr::probe {

namespace {

using hooks::Anchor;

// Engine layout from M3 (docs/ams2-renderer.md), AMS2AVX.exe build aa43bd68.
constexpr size_t kTaskCtx = 0x860;       // eye tasks (CBeginEye, CEndEye): render context pointer
constexpr size_t kTaskView = 0x868;      // eye tasks: view number (int)
constexpr size_t kCtxView = 0x27b04;     // render context: current view
constexpr size_t kCtxSlots = 0x27bc8;    // per view {camera, camera, main camera}, stride 0x18
constexpr size_t kCtxRt = 0x27c30;       // per view render target set, stride 8
constexpr size_t kCtxQuery = 0x70c;      // per view scene query (visibility list) index, stride 4
constexpr size_t kCtxEyeState = 0x180;   // CBeginEye: FUN_140db7630(view - 1)
constexpr size_t kCtxCamBlock = 0x190;   // per view camera data block, stride 0x1a0
constexpr size_t kCamBlockSize = 0x1a0;
constexpr size_t kSecCams[] = {0x2ca8, 0x2cc0, 0x2cd8};  // CInGameSection: left and right eye cameras, one more
constexpr int kViews = 3;

std::atomic<bool> g_on{false};
std::atomic<bool> g_fault{false};
bool g_engineHooked = false;
bool g_d3dHooked = false;

// Probe code runs in its own SEH guard: a fault stops the probe, the detours keep calling the originals.
template <class F>
void Safe(const char* where, F&& f) {
    if (g_fault.load(std::memory_order_relaxed)) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g_fault = true;
        Log("ERROR: exception 0x%08lX in the probe (%s). Probe stopped; hooks only pass calls on.", code, where);
    }
}

template <class T>
bool ReadAt(const void* base, size_t off, T& out) {
    const uint8_t* p = static_cast<const uint8_t*>(base) + off;
    if (!hooks::Readable(p, sizeof(T))) return false;
    memcpy(&out, p, sizeof(T));
    return true;
}

bool LooksLikePointer(uintptr_t v) { return v >= 0x10000 && v < 0x7FFFFFFF0000ull && (v & 7) == 0; }

std::string ModRva(const void* p) {
    HMODULE m = nullptr;
    char buf[160];
    if (p && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR)p, &m) && m) {
        wchar_t name[MAX_PATH] = {};
        GetModuleFileNameW(m, name, MAX_PATH);
        snprintf(buf, sizeof(buf), "%s+0x%llX", Narrow(fs::path(name).filename().wstring()).c_str(),
                 (unsigned long long)((uintptr_t)p - (uintptr_t)m));
    } else {
        snprintf(buf, sizeof(buf), "%p", p);
    }
    return buf;
}

// ---------------------------------------------------------------- D3D11 description helpers

std::string DescribeResource(ID3D11Resource* r) {
    char buf[200];
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    r->GetType(&dim);
    ID3D11Texture2D* t = nullptr;
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D && SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t))) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        t->Release();
        snprintf(buf, sizeof(buf), "tex2d %p %ux%u fmt %d samples %u mips %u array %u bind 0x%X misc 0x%X", (void*)r,
                 d.Width, d.Height, (int)d.Format, d.SampleDesc.Count, d.MipLevels, d.ArraySize, d.BindFlags, d.MiscFlags);
    } else {
        snprintf(buf, sizeof(buf), "resource %p dimension %d", (void*)r, (int)dim);
    }
    return buf;
}

std::string DescribeResourcePtr(void* p) {
    if (!p) return "null";
    std::string s = "(not a D3D11 resource)";
    ID3D11Resource* r = nullptr;
    if (SUCCEEDED(static_cast<IUnknown*>(p)->QueryInterface(__uuidof(ID3D11Resource), (void**)&r)) && r) {
        s = DescribeResource(r);
        r->Release();
    }
    return s;
}

HMODULE D3D11Module() {
    static HMODULE m = GetModuleHandleW(L"d3d11.dll");
    return m;
}

// A D3D11 object is recognized by its vtable inside d3d11.dll; only then QueryInterface is called on it.
bool IsD3D11Object(uintptr_t p) {
    if (!LooksLikePointer(p) || !hooks::Readable((void*)p, 8)) return false;
    uintptr_t vt = *(uintptr_t*)p;
    if (!LooksLikePointer(vt)) return false;
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)vt, &m))
        return false;
    return m && m == D3D11Module();
}

std::string DescribeD3D11Object(uintptr_t p) {
    IUnknown* u = (IUnknown*)p;
    ID3D11Resource* r = nullptr;
    if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11Resource), (void**)&r)) && r) {
        std::string s = DescribeResource(r);
        r->Release();
        return s;
    }
    ID3D11View* v = nullptr;
    if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11View), (void**)&v)) && v) {
        const char* kind = "view";
        IUnknown* t = nullptr;
        if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11RenderTargetView), (void**)&t))) kind = "RTV";
        else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11ShaderResourceView), (void**)&t))) kind = "SRV";
        else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11DepthStencilView), (void**)&t))) kind = "DSV";
        else if (SUCCEEDED(u->QueryInterface(__uuidof(ID3D11UnorderedAccessView), (void**)&t))) kind = "UAV";
        if (t) t->Release();
        ID3D11Resource* res = nullptr;
        v->GetResource(&res);
        v->Release();
        char buf[64];
        snprintf(buf, sizeof(buf), "%s %p of ", kind, (void*)p);
        std::string s = buf + (res ? DescribeResource(res) : std::string("null"));
        if (res) res->Release();
        return s;
    }
    char buf[96];
    snprintf(buf, sizeof(buf), "d3d11 object %p (vtable %s)", (void*)p, ModRva(*(void**)p).c_str());
    return buf;
}

// ---------------------------------------------------------------- state

enum Ev : uint8_t {
    EvBeginEye, EvBeginEyeDone, EvEndEye, EvEndFrame, EvEndFrameDone, EvFunctor, EvFunctorDone, EvSubmit, EvSubmitDone,
    EvCopy
};
const char* const kEvNames[] = {"BeginEye", "BeginEye done", "EndEye", "EndFrame", "EndFrame done", "CameraFunctor",
                                "CameraFunctor done", "Submit", "Submit done", "Copy"};

struct Event {
    Ev kind;
    int view;
    DWORD thread;
    double t;
    uint64_t imm, other, clears;
    char note[200];
};

struct Built {
    void* task;
    int kind;  // 0 CBeginEye, 1 CEndEye, 2 CCaptureBonnetReflection
    int view;
};
const char* const kBuiltNames[] = {"CBeginEye", "CEndEye", "CCaptureBonnetReflection"};

struct Stats {
    uint64_t frames = 0, inGame = 0;
    uint64_t imm = 0, other = 0, offThread = 0, ecl = 0, copies = 0, clears = 0;  // all frames
    uint64_t eyeImm[kViews] = {}, eyeN[kViews] = {}, outsideImm = 0;   // in-game frames
    double eyeMs[kViews] = {};
    double beginMs[kViews] = {};      // CBeginEye::Execute (the game's own code inside the detour)
    double frameMs = 0, endFrameMs = 0;  // in-game frames: WaitGetPoses entry to the next one; CEndFrame::Execute
};

std::mutex g_m;
Built g_built[16];
int g_builtCount = 0;
int g_builderCalls = 0;
void* g_section = nullptr;
Event g_ev[128];
int g_evCount = 0;
bool g_evOverflow = false;
uint64_t g_frameNo = 0;
double g_frameT0 = 0;
uint64_t g_frameImm0 = 0, g_frameOther0 = 0, g_frameOff0 = 0, g_frameEcl0 = 0, g_frameCopies0 = 0, g_frameClears0 = 0;
bool g_frameInGame = false;
std::atomic<bool> g_detail{false};  // the current frame is logged in full (read without the lock by NoteCopy)
bool g_dumpPending = false;  // dump the render context at the next in-game CBeginEye of this frame
int g_detailLeft = 0;
bool g_periodicDetail = false;
int g_dumps = 0;
uint64_t g_inGameFrames = 0;
Stats g_st, g_tot;
struct CopySeen {
    void* dst;
    void* src;
};
CopySeen g_copySeen[16];
int g_copySeenCount = 0;
int g_copyOutLogs[2] = {};

// D3D11 (atomics: the detours run on the game's threads for every draw)
std::atomic<ID3D11DeviceContext*> g_imm{nullptr};
ID3D11Device* g_dev = nullptr;  // no reference held, only compared
std::atomic<uint64_t> g_immDraws{0}, g_otherDraws{0}, g_immOffThread{0}, g_ecl{0}, g_copies{0}, g_immClears{0};
std::atomic<DWORD> g_renderThread{0}, g_drawThread{0};
std::atomic<void*> g_submitTex[8] = {};  // recent Submit textures: [eye * 4 + i]
int g_submitTexNext[2] = {};
thread_local int t_inSubmit = -1;
thread_local bool t_inBuilder = false;

bool IsSubmitTex(void* p, int* eye) {
    if (!p) return false;
    for (int i = 0; i < 8; ++i) {
        if (g_submitTex[i].load(std::memory_order_relaxed) == p) {
            if (eye) *eye = i / 4;
            return true;
        }
    }
    return false;
}

void AddEventLocked(Ev kind, int view, const char* note) {
    if (g_evCount >= (int)std::size(g_ev)) {
        g_evOverflow = true;
        return;
    }
    Event& e = g_ev[g_evCount++];
    e.kind = kind;
    e.view = view;
    e.thread = GetCurrentThreadId();
    e.t = NowMs();
    e.imm = g_immDraws.load(std::memory_order_relaxed);
    e.other = g_otherDraws.load(std::memory_order_relaxed);
    e.clears = g_immClears.load(std::memory_order_relaxed);
    snprintf(e.note, sizeof(e.note), "%s", note ? note : "");
}

int BuiltIndexLocked(void* task) {
    for (int i = 0; i < g_builtCount; ++i)
        if (g_built[i].task == task) return i;
    return -1;
}

int TaskView(void* task) {
    int v = -1;
    ReadAt(task, kTaskView, v);
    return v;
}

void* TaskCtx(void* task) {
    void* c = nullptr;
    ReadAt(task, kTaskCtx, c);
    return c;
}

// ---------------------------------------------------------------- render context dump

void DumpFloats(const char* label, const void* p, size_t bytes) {
    if (!hooks::Readable(p, bytes)) {
        Log("  %s %p: not readable", label, p);
        return;
    }
    Log("  %s %p (0x%zX B, floats):", label, p, bytes);
    const float* f = static_cast<const float*>(p);
    for (size_t i = 0; i < bytes / 4; i += 8) {
        char line[256];
        int n = snprintf(line, sizeof(line), "    +0x%03zX:", i * 4);
        for (size_t k = i; k < i + 8 && k < bytes / 4; ++k) n += snprintf(line + n, sizeof(line) - n, " %.5g", f[k]);
        LogRaw(line);
    }
}

// D3D11 objects referenced from a render target set: directly (+off) or one pointer deeper (+off -> +off2).
void ScanForD3D11(const char* label, void* obj) {
    if (!LooksLikePointer((uintptr_t)obj) || !hooks::Readable(obj, 0x200)) {
        Log("  %s %p: not readable", label, obj);
        return;
    }
    int lines = 0;
    for (size_t off = 0; off < 0x200 && lines < 32; off += 8) {
        uintptr_t q = *(uintptr_t*)((uint8_t*)obj + off);
        if (IsD3D11Object(q)) {
            Log("  %s +0x%03zX: %s", label, off, DescribeD3D11Object(q).c_str());
            ++lines;
            continue;
        }
        if (!LooksLikePointer(q) || !hooks::Readable((void*)q, 0x100)) continue;
        for (size_t off2 = 0; off2 < 0x100 && lines < 32; off2 += 8) {
            uintptr_t r = *(uintptr_t*)(q + off2);
            if (IsD3D11Object(r)) {
                Log("  %s +0x%03zX -> +0x%02zX: %s", label, off, off2, DescribeD3D11Object(r).c_str());
                ++lines;
            }
        }
    }
    if (!lines) Log("  %s %p: no D3D11 objects within two levels", label, obj);
}

void DumpContextLocked(void* ctx, bool full) {
    int view = -1;
    void* eyeState = nullptr;
    ReadAt(ctx, kCtxView, view);
    ReadAt(ctx, kCtxEyeState, eyeState);
    Log("Probe dump %d (frame %llu, thread %lu): render context %p, current view %d, +0x180 %p", g_dumps,
        (unsigned long long)g_frameNo, GetCurrentThreadId(), ctx, view, eyeState);
    void* cams[kViews * 3] = {};
    for (int v = 0; v < kViews; ++v) {
        void* rt = nullptr;
        int query = -1;
        ReadAt(ctx, kCtxSlots + v * 0x18, cams[v * 3]);
        ReadAt(ctx, kCtxSlots + v * 0x18 + 8, cams[v * 3 + 1]);
        ReadAt(ctx, kCtxSlots + v * 0x18 + 16, cams[v * 3 + 2]);
        ReadAt(ctx, kCtxRt + v * 8, rt);
        ReadAt(ctx, kCtxQuery + v * 4, query);
        Log("  view %d: slot {%p, %p, %p}, render targets %p, scene query %d", v, cams[v * 3], cams[v * 3 + 1],
            cams[v * 3 + 2], rt, query);
    }
    if (g_section) {
        char line[256];
        int n = snprintf(line, sizeof(line), "  section %p:", g_section);
        for (size_t off : kSecCams) {
            void* c = nullptr;
            ReadAt(g_section, off, c);
            n += snprintf(line + n, sizeof(line) - n, " +0x%zX %p", off, c);
        }
        Log("%s", line);
    }
    void* seen[kViews * 3] = {};
    int nSeen = 0;
    for (void* c : cams) {
        if (!c || std::find(seen, seen + nSeen, c) != seen + nSeen) continue;
        seen[nSeen++] = c;
        void* vt = nullptr;
        ReadAt(c, 0, vt);
        Log("  camera %p: vtable %s", c, ModRva(vt).c_str());
    }
    for (int v = 0; v < kViews; ++v) {
        void* rt = nullptr;
        ReadAt(ctx, kCtxRt + v * 8, rt);
        char label[48];
        snprintf(label, sizeof(label), "render targets[%d]", v);
        ScanForD3D11(label, rt);
    }
    if (!full) return;
    for (int v = 0; v < kViews; ++v) {
        char label[48];
        snprintf(label, sizeof(label), "camera block[%d] (ctx+0x%zX)", v, kCtxCamBlock + v * kCamBlockSize);
        DumpFloats(label, (uint8_t*)ctx + kCtxCamBlock + v * kCamBlockSize, kCamBlockSize);
    }
    for (int i = 0; i < nSeen; ++i) DumpFloats("camera object", seen[i], 0x400);
}

// ---------------------------------------------------------------- frame bookkeeping

void FinishFrameLocked() {
    uint64_t imm = g_immDraws.load() - g_frameImm0, other = g_otherDraws.load() - g_frameOther0;
    ++g_st.frames;
    g_st.imm += imm;
    g_st.other += other;
    g_st.offThread += g_immOffThread.load() - g_frameOff0;
    g_st.ecl += g_ecl.load() - g_frameEcl0;
    g_st.copies += g_copies.load() - g_frameCopies0;
    g_st.clears += g_immClears.load() - g_frameClears0;
    if (g_frameInGame) {
        ++g_st.inGame;
        ++g_inGameFrames;
        uint64_t inEyes = 0;
        for (int i = 0; i < g_evCount; ++i) {
            const Event& b = g_ev[i];
            if (b.kind != EvBeginEye || b.view < 0 || b.view >= kViews) continue;
            for (int k = i + 1; k < g_evCount; ++k) {
                const Event& e = g_ev[k];
                if (e.kind == EvEndEye && e.view == b.view) {
                    g_st.eyeImm[b.view] += e.imm - b.imm;
                    g_st.eyeMs[b.view] += e.t - b.t;
                    ++g_st.eyeN[b.view];
                    inEyes += e.imm - b.imm;
                    break;
                }
            }
        }
        g_st.outsideImm += imm > inEyes ? imm - inEyes : 0;
        g_st.frameMs += NowMs() - g_frameT0;
        for (int i = 0; i + 1 < g_evCount; ++i) {
            const Event& b = g_ev[i];
            const Event& n = g_ev[i + 1];
            if (b.kind == EvBeginEye && n.kind == EvBeginEyeDone && b.view >= 0 && b.view < kViews && b.thread == n.thread)
                g_st.beginMs[b.view] += n.t - b.t;
            if (b.kind == EvEndFrame && n.kind == EvEndFrameDone) g_st.endFrameMs += n.t - b.t;
        }
    }
    if (!g_detail) return;
    Log("Probe frame %llu (%s, %d events%s, %.2f ms, immediate context draws %llu, other contexts %llu):",
        (unsigned long long)g_frameNo, g_frameInGame ? "in-game" : "not in-game", g_evCount,
        g_evOverflow ? ", list full" : "", NowMs() - g_frameT0, (unsigned long long)imm, (unsigned long long)other);
    uint64_t prevImm = g_frameImm0, prevOther = g_frameOther0, prevClears = g_frameClears0;
    for (int i = 0; i < g_evCount; ++i) {
        const Event& e = g_ev[i];
        char line[512];
        snprintf(line, sizeof(line), "  %8.3f ms  thread %5lu  %-18s view %2d  draws +%-5llu clears +%-3llu other +%-4llu %s",
                 e.t - g_frameT0, e.thread, kEvNames[e.kind], e.view, (unsigned long long)(e.imm - prevImm),
                 (unsigned long long)(e.clears - prevClears), (unsigned long long)(e.other - prevOther), e.note);
        LogRaw(line);
        prevImm = e.imm;
        prevOther = e.other;
        prevClears = e.clears;
    }
}

void StartFrameLocked() {
    ++g_frameNo;
    g_frameT0 = NowMs();
    g_evCount = 0;
    g_evOverflow = false;
    g_frameInGame = false;
    g_frameImm0 = g_immDraws.load();
    g_frameOther0 = g_otherDraws.load();
    g_frameOff0 = g_immOffThread.load();
    g_frameEcl0 = g_ecl.load();
    g_frameCopies0 = g_copies.load();
    g_frameClears0 = g_immClears.load();
    g_detail = false;
    g_dumpPending = false;
    if (g_inGameFrames > 0 && (g_detailLeft > 0 || g_periodicDetail)) {
        g_detail = true;
        g_dumpPending = true;
        if (g_detailLeft > 0)
            --g_detailLeft;
        else
            g_periodicDetail = false;
    }
}

// ---------------------------------------------------------------- engine detours

using Fn4 = void*(__fastcall*)(void*, void*, void*, void*);
Fn4 o_builder = nullptr, o_beginCtor = nullptr, o_endCtor = nullptr, o_bonnetCtor = nullptr;
Fn4 o_beginExec = nullptr, o_endExec = nullptr, o_endFrameExec = nullptr, o_functor = nullptr;

void* __fastcall D_Builder(void* a, void* b, void* c, void* d) {
    Safe("builder", [&] {
        std::lock_guard lock(g_m);
        ++g_builderCalls;
        g_builtCount = 0;
        g_section = a;
        Log("Probe: BuildRenderTasksOculusRift call %d (section %p, thread %lu)", g_builderCalls, a, GetCurrentThreadId());
    });
    t_inBuilder = true;
    void* r = o_builder(a, b, c, d);
    t_inBuilder = false;
    Safe("builder (after)", [&] {
        std::lock_guard lock(g_m);
        std::string s;
        for (int i = 0; i < g_builtCount; ++i) {
            char buf[96];
            snprintf(buf, sizeof(buf), "%s%s(%d) %p", s.empty() ? "" : ", ", kBuiltNames[g_built[i].kind], g_built[i].view,
                     g_built[i].task);
            s += buf;
        }
        Log("Probe: builder done, eye tasks: %s", s.empty() ? "(none)" : s.c_str());
    });
    return r;
}

void NoteBuilt(void* task, int kind, void* viewArg) {
    if (!t_inBuilder) return;
    Safe("ctor", [&] {
        std::lock_guard lock(g_m);
        if (g_builtCount < (int)std::size(g_built)) g_built[g_builtCount++] = {task, kind, (int)(intptr_t)viewArg};
    });
}

void* __fastcall D_BeginCtor(void* a, void* b, void* c, void* d) {
    void* r = o_beginCtor(a, b, c, d);
    NoteBuilt(a, 0, c);
    return r;
}

void* __fastcall D_EndCtor(void* a, void* b, void* c, void* d) {
    void* r = o_endCtor(a, b, c, d);
    NoteBuilt(a, 1, c);
    return r;
}

void* __fastcall D_BonnetCtor(void* a, void* b, void* c, void* d) {
    void* r = o_bonnetCtor(a, b, c, d);
    NoteBuilt(a, 2, c);
    return r;
}

void* __fastcall D_BeginExec(void* task, void* b, void* c, void* d) {
    Safe("CBeginEye", [&] {
        std::lock_guard lock(g_m);
        bool vr = BuiltIndexLocked(task) >= 0;
        if (vr && !g_frameInGame) {
            g_frameInGame = true;
            if (g_inGameFrames == 0 && g_st.inGame == 0 && g_tot.inGame == 0)
                Log("Probe: first in-game frame (frame %llu, CBeginEye on thread %lu)", (unsigned long long)g_frameNo,
                    GetCurrentThreadId());
        }
        char note[96];
        snprintf(note, sizeof(note), "%s task %p", vr ? "VR graph" : "other graph", task);
        AddEventLocked(EvBeginEye, TaskView(task), note);
    });
    void* r = o_beginExec(task, b, c, d);
    Safe("CBeginEye (after)", [&] {
        std::lock_guard lock(g_m);
        void* ctx = TaskCtx(task);
        void* eyeState = nullptr;
        if (ctx) ReadAt(ctx, kCtxEyeState, eyeState);
        char note[96];
        snprintf(note, sizeof(note), "ctx %p, +0x180 %p", ctx, eyeState);
        AddEventLocked(EvBeginEyeDone, TaskView(task), note);
        if (g_dumpPending && ctx && BuiltIndexLocked(task) >= 0) {
            g_dumpPending = false;
            DumpContextLocked(ctx, g_dumps == 0);
            ++g_dumps;
        }
    });
    return r;
}

void* __fastcall D_EndExec(void* task, void* b, void* c, void* d) {
    Safe("CEndEye", [&] {
        std::lock_guard lock(g_m);
        char note[200] = "";
        ID3D11DeviceContext* imm = g_imm.load();
        // The immediate context is not thread-safe: read its bindings only on the thread that draws on it.
        if (g_detail && imm && GetCurrentThreadId() == g_drawThread.load()) {
            ID3D11RenderTargetView* rtv[8] = {};
            ID3D11DepthStencilView* dsv = nullptr;
            imm->OMGetRenderTargets(8, rtv, &dsv);
            int n = 0;
            for (auto* v : rtv) n += v != nullptr;
            std::string s = "no render target";
            if (rtv[0]) {
                ID3D11Resource* res = nullptr;
                rtv[0]->GetResource(&res);
                if (res) {
                    s = "RT0 " + DescribeResource(res);
                    res->Release();
                }
            }
            snprintf(note, sizeof(note), "%d RTV, %s%s", n, s.c_str(), dsv ? ", DSV bound" : "");
            for (auto* v : rtv)
                if (v) v->Release();
            if (dsv) dsv->Release();
        } else if (g_detail) {
            snprintf(note, sizeof(note), "(bindings not read: not the drawing thread %lu)", g_drawThread.load());
        }
        AddEventLocked(EvEndEye, TaskView(task), note);
    });
    return o_endExec(task, b, c, d);
}

void* __fastcall D_EndFrameExec(void* task, void* b, void* c, void* d) {
    Safe("CEndFrame", [&] {
        std::lock_guard lock(g_m);
        AddEventLocked(EvEndFrame, -1, "");
    });
    void* r = o_endFrameExec(task, b, c, d);
    Safe("CEndFrame (after)", [&] {
        std::lock_guard lock(g_m);
        AddEventLocked(EvEndFrameDone, -1, "");
    });
    return r;
}

void* __fastcall D_Functor(void* a, void* b, void* c, void* d) {
    Safe("camera functor", [&] {
        std::lock_guard lock(g_m);
        char note[64];
        snprintf(note, sizeof(note), "section %p", a);
        AddEventLocked(EvFunctor, -1, note);
    });
    void* r = o_functor(a, b, c, d);
    Safe("camera functor (after)", [&] {
        std::lock_guard lock(g_m);
        AddEventLocked(EvFunctorDone, -1, "");
    });
    return r;
}

// ---------------------------------------------------------------- D3D11 detours

using DrawIndexedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, INT);
using DrawFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT);
using DrawIndexedInstancedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using DrawInstancedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
using DispatchFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT);
using CopySubresourceRegionFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, UINT, UINT, UINT,
                                                         ID3D11Resource*, UINT, const D3D11_BOX*);
using CopyResourceFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, ID3D11Resource*);
using ExecuteCommandListFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11CommandList*, BOOL);
using ClearRtvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11RenderTargetView*, const FLOAT*);
using ClearDsvFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);

DrawIndexedFn o_drawIndexed = nullptr;
DrawFn o_draw = nullptr;
DrawIndexedInstancedFn o_drawIndexedInstanced = nullptr;
DrawInstancedFn o_drawInstanced = nullptr;
DispatchFn o_dispatch = nullptr;
CopySubresourceRegionFn o_copyRegion = nullptr;
CopyResourceFn o_copyResource = nullptr;
ExecuteCommandListFn o_ecl = nullptr;
ClearRtvFn o_clearRtv = nullptr;
ClearDsvFn o_clearDsv = nullptr;

inline void CountDraw(ID3D11DeviceContext* c) {
    if (c == g_imm.load(std::memory_order_relaxed)) {
        g_immDraws.fetch_add(1, std::memory_order_relaxed);
        DWORD t = GetCurrentThreadId();
        g_drawThread.store(t, std::memory_order_relaxed);
        if (t != g_renderThread.load(std::memory_order_relaxed)) g_immOffThread.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_otherDraws.fetch_add(1, std::memory_order_relaxed);
    }
}

void STDMETHODCALLTYPE D_DrawIndexed(ID3D11DeviceContext* c, UINT a, UINT b, INT d) {
    CountDraw(c);
    o_drawIndexed(c, a, b, d);
}
void STDMETHODCALLTYPE D_Draw(ID3D11DeviceContext* c, UINT a, UINT b) {
    CountDraw(c);
    o_draw(c, a, b);
}
void STDMETHODCALLTYPE D_DrawIndexedInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, INT e, UINT f) {
    CountDraw(c);
    o_drawIndexedInstanced(c, a, b, d, e, f);
}
void STDMETHODCALLTYPE D_DrawInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, UINT e) {
    CountDraw(c);
    o_drawInstanced(c, a, b, d, e);
}
void STDMETHODCALLTYPE D_Dispatch(ID3D11DeviceContext* c, UINT a, UINT b, UINT d) {
    CountDraw(c);
    o_dispatch(c, a, b, d);
}

// Copies into or out of Submit textures are logged once per (destination, source) pair; in a full frame every copy
// on the immediate context goes into the event list.
void NoteCopy(ID3D11DeviceContext* c, ID3D11Resource* dst, ID3D11Resource* src, bool region) {
    if (c != g_imm.load(std::memory_order_relaxed)) return;
    g_copies.fetch_add(1, std::memory_order_relaxed);
    int eDst = -1, eSrc = -1;
    bool toSubmit = IsSubmitTex(dst, &eDst), fromSubmit = IsSubmitTex(src, &eSrc);
    if (!toSubmit && !fromSubmit && !g_detail.load(std::memory_order_relaxed)) return;
    Safe("copy", [&] {
        std::lock_guard lock(g_m);
        const char* where = t_inSubmit >= 0 ? " (inside Submit)" : "";
        // SteamVR copies out of the Submit textures itself (inside Submit and later on the same thread) into textures
        // that change; only the first two such copies per eye are logged, copies into Submit textures all.
        int e = toSubmit ? eDst : eSrc;
        bool quiet = !toSubmit && fromSubmit && ++g_copyOutLogs[e] > 2;
        if ((toSubmit || fromSubmit) && !quiet) {
            bool seen = false;
            for (int i = 0; i < g_copySeenCount; ++i) seen |= g_copySeen[i].dst == dst && g_copySeen[i].src == src;
            if (!seen && g_copySeenCount < (int)std::size(g_copySeen)) {
                g_copySeen[g_copySeenCount++] = {dst, src};
                Log("Probe: %s %s Submit texture of the %s eye%s, thread %lu: %s <- %s",
                    region ? "CopySubresourceRegion" : "CopyResource", toSubmit ? "into" : "out of", e ? "right" : "left",
                    where, GetCurrentThreadId(), DescribeResource(dst).c_str(), DescribeResource(src).c_str());
            }
        }
        if (g_detail) {
            char note[200];
            snprintf(note, sizeof(note), "%s%s dst %p%s <- src %p%s", region ? "region" : "resource", where, (void*)dst,
                     toSubmit ? " (Submit tex)" : "", (void*)src, fromSubmit ? " (Submit tex)" : "");
            AddEventLocked(EvCopy, -1, note);
        }
    });
}

void STDMETHODCALLTYPE D_CopySubresourceRegion(ID3D11DeviceContext* c, ID3D11Resource* dst, UINT ds, UINT x, UINT y,
                                               UINT z, ID3D11Resource* src, UINT ss, const D3D11_BOX* box) {
    NoteCopy(c, dst, src, true);
    o_copyRegion(c, dst, ds, x, y, z, src, ss, box);
}
void STDMETHODCALLTYPE D_CopyResource(ID3D11DeviceContext* c, ID3D11Resource* dst, ID3D11Resource* src) {
    NoteCopy(c, dst, src, false);
    o_copyResource(c, dst, src);
}
void STDMETHODCALLTYPE D_ClearRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT* color) {
    if (c == g_imm.load(std::memory_order_relaxed)) g_immClears.fetch_add(1, std::memory_order_relaxed);
    o_clearRtv(c, v, color);
}
void STDMETHODCALLTYPE D_ClearDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT flags, FLOAT depth, UINT8 stencil) {
    if (c == g_imm.load(std::memory_order_relaxed)) g_immClears.fetch_add(1, std::memory_order_relaxed);
    o_clearDsv(c, v, flags, depth, stencil);
}
void STDMETHODCALLTYPE D_ExecuteCommandList(ID3D11DeviceContext* c, ID3D11CommandList* l, BOOL restore) {
    if (c == g_imm.load(std::memory_order_relaxed)) g_ecl.fetch_add(1, std::memory_order_relaxed);
    o_ecl(c, l, restore);
}

// ID3D11DeviceContext vtable slots (d3d11.h order).
constexpr int kSlotDrawIndexed = 12, kSlotDraw = 13, kSlotDrawIndexedInstanced = 20, kSlotDrawInstanced = 21,
              kSlotDispatch = 41, kSlotCopySubresourceRegion = 46, kSlotCopyResource = 47, kSlotClearRtv = 50,
              kSlotClearDsv = 53, kSlotExecuteCommandList = 58;

void HookD3D11(ID3D11DeviceContext* imm) {
    void** vt = *(void***)imm;
    bool ok = true;
    ok &= hooks::CreateAt(vt[kSlotDrawIndexed], (void*)D_DrawIndexed, (void**)&o_drawIndexed, "DrawIndexed");
    ok &= hooks::CreateAt(vt[kSlotDraw], (void*)D_Draw, (void**)&o_draw, "Draw");
    ok &= hooks::CreateAt(vt[kSlotDrawIndexedInstanced], (void*)D_DrawIndexedInstanced, (void**)&o_drawIndexedInstanced,
                          "DrawIndexedInstanced");
    ok &= hooks::CreateAt(vt[kSlotDrawInstanced], (void*)D_DrawInstanced, (void**)&o_drawInstanced, "DrawInstanced");
    ok &= hooks::CreateAt(vt[kSlotDispatch], (void*)D_Dispatch, (void**)&o_dispatch, "Dispatch");
    ok &= hooks::CreateAt(vt[kSlotCopySubresourceRegion], (void*)D_CopySubresourceRegion, (void**)&o_copyRegion,
                          "CopySubresourceRegion");
    ok &= hooks::CreateAt(vt[kSlotCopyResource], (void*)D_CopyResource, (void**)&o_copyResource, "CopyResource");
    ok &= hooks::CreateAt(vt[kSlotClearRtv], (void*)D_ClearRtv, (void**)&o_clearRtv, "ClearRenderTargetView");
    ok &= hooks::CreateAt(vt[kSlotClearDsv], (void*)D_ClearDsv, (void**)&o_clearDsv, "ClearDepthStencilView");
    ok &= hooks::CreateAt(vt[kSlotExecuteCommandList], (void*)D_ExecuteCommandList, (void**)&o_ecl, "ExecuteCommandList");
    BOOL protectedMt = FALSE;
    ID3D11Multithread* mt = nullptr;
    if (SUCCEEDED(imm->QueryInterface(__uuidof(ID3D11Multithread), (void**)&mt)) && mt) {
        protectedMt = mt->GetMultithreadProtected();
        mt->Release();
    }
    Log("Probe: D3D11 hooks %s on immediate context %p (type %d, multithread protected %d, DrawIndexed at %s)",
        ok ? "installed" : "INCOMPLETE", (void*)imm, (int)imm->GetType(), protectedMt, ModRva(vt[kSlotDrawIndexed]).c_str());
}

void LogStatsLocked(const Stats& s, const char* what) {
    double f = s.frames ? (double)s.frames : 1, g = s.inGame ? (double)s.inGame : 1;
    auto avg = [](uint64_t v, uint64_t n) { return n ? (double)v / n : 0.0; };
    Log("Probe stats%s: %llu frames (%llu in-game); per frame: immediate context draws %.1f (off the render thread %.1f), "
        "other contexts %.1f, ExecuteCommandList %.2f, copies %.1f, clears %.1f",
        what, (unsigned long long)s.frames, (unsigned long long)s.inGame, s.imm / f, s.offThread / f, s.other / f,
        s.ecl / f, s.copies / f, s.clears / f);
    if (s.inGame)
        Log("Probe stats%s: per in-game frame: immediate draws between CBeginEye and CEndEye: view 1 %.0f (%.2f ms), view 2 "
            "%.0f (%.2f ms); outside the eyes %.0f",
            what, avg(s.eyeImm[1], s.eyeN[1]), s.eyeN[1] ? s.eyeMs[1] / s.eyeN[1] : 0.0, avg(s.eyeImm[2], s.eyeN[2]),
            s.eyeN[2] ? s.eyeMs[2] / s.eyeN[2] : 0.0, s.outsideImm / g);
    if (s.inGame)
        Log("Probe stats%s: per in-game frame: %.2f ms (%.1f/s); CBeginEye::Execute view 1 %.2f ms, view 2 %.2f ms; "
            "CEndFrame::Execute %.2f ms",
            what, s.frameMs / g, s.frameMs > 0 ? 1000.0 * s.inGame / s.frameMs : 0.0,
            s.eyeN[1] ? s.beginMs[1] / s.eyeN[1] : 0.0, s.eyeN[2] ? s.beginMs[2] / s.eyeN[2] : 0.0, s.endFrameMs / g);
}

void AddStats(Stats& a, const Stats& b) {
    a.frames += b.frames;
    a.inGame += b.inGame;
    a.imm += b.imm;
    a.other += b.other;
    a.offThread += b.offThread;
    a.ecl += b.ecl;
    a.copies += b.copies;
    a.clears += b.clears;
    a.outsideImm += b.outsideImm;
    for (int v = 0; v < kViews; ++v) {
        a.eyeImm[v] += b.eyeImm[v];
        a.eyeN[v] += b.eyeN[v];
        a.eyeMs[v] += b.eyeMs[v];
        a.beginMs[v] += b.beginMs[v];
    }
    a.frameMs += b.frameMs;
    a.endFrameMs += b.endFrameMs;
}

}  // namespace

void Install() {
    if (Cfg().quadMode != QuadMode::Probe || !Cfg().wrap) return;
    if (!hooks::Init()) return;
    static bool first = true;
    if (!first) {
        hooks::EnableAll();
        g_on = true;
        return;
    }
    first = false;
    Safe("Install", [&] {
        if (!Cfg().probeEngine) {
            Log("Probe: engine hooks off (probe_engine=0)");
        } else if (hooks::EngineAvailable()) {
            bool ok = true;
            ok &= hooks::Create(Anchor::BuilderOculusRift, (void*)D_Builder, (void**)&o_builder);
            ok &= hooks::Create(Anchor::BeginEyeCtor, (void*)D_BeginCtor, (void**)&o_beginCtor);
            ok &= hooks::Create(Anchor::EndEyeCtor, (void*)D_EndCtor, (void**)&o_endCtor);
            ok &= hooks::Create(Anchor::BonnetReflectionCtor, (void*)D_BonnetCtor, (void**)&o_bonnetCtor);
            ok &= hooks::Create(Anchor::BeginEyeExecute, (void*)D_BeginExec, (void**)&o_beginExec);
            ok &= hooks::Create(Anchor::EndEyeExecute, (void*)D_EndExec, (void**)&o_endExec);
            ok &= hooks::Create(Anchor::EndFrameExecute, (void*)D_EndFrameExec, (void**)&o_endFrameExec);
            ok &= hooks::Create(Anchor::ViewCameraFunctor, (void*)D_Functor, (void**)&o_functor);
            g_engineHooked = true;
            Log("Probe: engine hooks %s (builder, eye task ctors and Execute, CEndFrame, camera functor)",
                ok ? "installed" : "INCOMPLETE");
        } else {
            Log("Probe: engine hooks off (exe not supported, see the Hooks lines); D3D11 counters only");
        }
        std::lock_guard lock(g_m);
        g_detailLeft = Cfg().probeFrames;
    });
    g_on = true;
}

void OnFrameStart() {
    g_renderThread.store(GetCurrentThreadId(), std::memory_order_relaxed);
    if (!g_on) return;
    Safe("frame", [&] {
        std::lock_guard lock(g_m);
        if (g_frameNo > 0) FinishFrameLocked();
        StartFrameLocked();
    });
}

void OnSubmitBegin(vr::EVREye eye, const vr::Texture_t* tex) {
    if (!g_on) return;
    Safe("Submit", [&] {
        int e = eye == vr::Eye_Right ? 1 : 0;
        if (tex && tex->eType == vr::API_DirectX && tex->handle && !IsSubmitTex(tex->handle, nullptr)) {
            g_submitTex[e * 4 + g_submitTexNext[e]].store(tex->handle);
            g_submitTexNext[e] = (g_submitTexNext[e] + 1) % 4;
            ID3D11Texture2D* t = nullptr;
            if (SUCCEEDED(static_cast<IUnknown*>(tex->handle)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
                ID3D11Device* dev = nullptr;
                t->GetDevice(&dev);
                if (dev && dev != g_dev) {
                    ID3D11DeviceContext* imm = nullptr;
                    dev->GetImmediateContext(&imm);
                    g_dev = dev;
                    g_imm = imm;  // no reference kept: the game owns the device for the whole session
                    if (imm) {
                        if (!g_d3dHooked) {
                            if (Cfg().probeD3D)
                                HookD3D11(imm);
                            else
                                Log("Probe: D3D11 hooks off (probe_d3d=0)");
                            // Frames before the first Submit (track loading) have no D3D11 data: log the next
                            // probe_frames in full again.
                            std::lock_guard lock(g_m);
                            g_detailLeft = Cfg().probeFrames;
                        }
                        g_d3dHooked = true;
                        imm->Release();
                    }
                }
                if (dev) dev->Release();
                t->Release();
            }
        }
        std::lock_guard lock(g_m);
        if (!g_detail) return;
        char note[200];
        snprintf(note, sizeof(note), "%s eye, %s", e ? "right" : "left",
                 tex && tex->eType == vr::API_DirectX ? DescribeResourcePtr(tex->handle).c_str() : "not D3D11");
        AddEventLocked(EvSubmit, e, note);
    });
    t_inSubmit = eye == vr::Eye_Right ? 1 : 0;
}

void OnSubmitEnd(vr::EVREye eye) {
    t_inSubmit = -1;
    if (!g_on) return;
    Safe("Submit (after)", [&] {
        std::lock_guard lock(g_m);
        if (g_detail) AddEventLocked(EvSubmitDone, eye == vr::Eye_Right ? 1 : 0, "");
    });
}

void LogStats(bool totals) {
    if (!g_on && !totals) return;
    Safe("LogStats", [&] {
        std::lock_guard lock(g_m);
        if (!g_st.frames && !g_tot.frames) return;
        AddStats(g_tot, g_st);
        if (totals) {
            LogStatsLocked(g_tot, " (session total)");
        } else {
            LogStatsLocked(g_st, "");
            g_periodicDetail = true;  // the next in-game frame is logged in full again
        }
        g_st = Stats{};
    });
}

void Shutdown() {
    if (!g_on) return;
    g_on = false;
    hooks::DisableAll();
    Safe("Shutdown", [&] {
        std::lock_guard lock(g_m);
        g_imm = nullptr;
        g_dev = nullptr;
        for (auto& t : g_submitTex) t = nullptr;
        g_frameNo = 0;
        g_inGameFrames = 0;
        g_st = g_tot = Stats{};
        g_copySeenCount = 0;
        g_copyOutLogs[0] = g_copyOutLogs[1] = 0;
        g_detailLeft = Cfg().probeFrames;
        g_dumps = 0;
    });
}

}  // namespace qvfr::probe
