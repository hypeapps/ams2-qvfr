#include "taa_views.h"

#include <windows.h>

#include <d3d11_1.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "dxbc.h"
#include "gpu.h"
#include "hooks.h"
#include "sysinfo.h"

namespace qvfr::taaviews {
namespace {

using hooks::Anchor;

// CTAAControl (docs/ams2-renderer.md, "TAA i bufor prędkości"). The exe hash check makes these offsets valid.
constexpr uintptr_t kTaaVtableRva = 0x21163c8;
constexpr size_t kDevTaa = 0xe450;  // engine device -> CTAAControl
constexpr int kSlot = 2;            // view slot of outer R and of both focus groups
constexpr int kArrays = 6;
// Per-view target arrays (3 slots each): render target objects and, in the same order, their textures.
constexpr size_t kRtArr[kArrays] = {0x68, 0x80, 0x98, 0xb0, 0xc8, 0xe0};
constexpr size_t kTexArr[kArrays] = {0x150, 0x168, 0x180, 0x198, 0x1b0, 0x1c8};
constexpr const char* kArrName[kArrays] = {"history A", "history B", "velocity", "previous depth", "locks A",
                                           "locks B"};
// Previous camera matrices per view slot (64 B each), written at the end of the resolve.
constexpr int kMats = 3;
constexpr size_t kMatArr[kMats] = {0x13f0, 0x14b0, 0x1570};
constexpr size_t kMatSize = 0x40;
// Engine names of the copies (the helper copies the string).
constexpr const char* kCopyName[2][kArrays] = {
    {"qvfrTaaFocusL_HistoryA", "qvfrTaaFocusL_HistoryB", "qvfrTaaFocusL_Velocity", "qvfrTaaFocusL_PrevDepth",
     "qvfrTaaFocusL_LocksA", "qvfrTaaFocusL_LocksB"},
    {"qvfrTaaFocusR_HistoryA", "qvfrTaaFocusR_HistoryB", "qvfrTaaFocusR_Velocity", "qvfrTaaFocusR_PrevDepth",
     "qvfrTaaFocusR_LocksA", "qvfrTaaFocusR_LocksB"}};
constexpr const char* kEyeName[2] = {"focus L", "focus R"};

// 0x16724b0: creates one target (texture + render target object) unless *rt is already set; true on success.
using CreateFn = bool(__fastcall*)(void* taa, uint32_t w, uint32_t h, const char* name, int kind, void** rt,
                                   void** tex);
// 0x16729a0 (vtable slot 32): releases all of the TAA's targets (the destructor calls it).
using ReleaseAllFn = void(__fastcall*)(void* taa);
// 0x1672910: releases the 3 entries of one array pair through the engine device (rcx unused).
using ReleaseArrayFn = void(__fastcall*)(void* unused, void** rt3, void** tex3);

CreateFn o_create = nullptr;
ReleaseAllFn o_releaseAll = nullptr;
ReleaseArrayFn g_releaseArray = nullptr;
void** g_devGlobal = nullptr;
bool g_installed = false;
std::atomic<bool> g_dead{false};
bool g_enabled = false;  // render thread
thread_local bool t_inside = false;

// This module's copies for one CTAAControl. Entries in 3-slot arrays (only [0] used) for the release helper.
struct Copies {
    void* taa = nullptr;
    void* rt[2][kArrays][3] = {};
    void* tex[2][kArrays][3] = {};
    int made = 0;  // array copies made (both eyes)
    uint32_t w = 0, h = 0;
    uint8_t mat[2][kMats][kMatSize] = {};
    bool haveMat[2] = {};
};
Copies g_c;

struct Swap {
    bool on = false;
    uint8_t* taa = nullptr;
    int eye = 0;
    void* rt[kArrays] = {};
    void* tex[kArrays] = {};
    uint8_t mat[kMats][kMatSize] = {};  // outer R's, put back after the group
    uint8_t matIn[kMats][kMatSize] = {};  // what the group started with (a change = the resolve ran)
};
Swap g_sw;

struct Stats {
    uint64_t swaps[2] = {};
    // Groups whose slot 2 matrices changed inside the group: the engine's resolve wrote them, so the focus groups do
    // use view slot 2 (with a still head the matrices may stay the same).
    uint64_t resolved[2] = {};
    uint64_t noCopies = 0;      // focus groups with a TAA but without this module's copies
    uint64_t changed = 0;       // groups after which a swapped field held something else
    uint64_t late = 0;          // swaps put back at frame start
};
Stats g_st, g_tot;
std::mutex g_m;
char g_state[160] = "off";
bool g_beforeLogged = false, g_offLogged = false;

// Diagnostics: in a shot frame the full-screen draws of outer R and focus L are logged with what is bound.
constexpr int kDumpDraws = 160;
bool g_dumpFrame = false;           // render thread
std::atomic<int> g_dumping{-1};     // group being dumped
std::atomic<bool> g_fixOn{false};  // TAA viewportScale fix (see FixDraw)
// Object motion vectors in a focus group (see FixObjectVelocity): this focus view's projection and view-projection
// of the last frame, from its TAA state. Render thread.
struct PrevFocus {
    bool on = false;
    float proj[16] = {};
    float viewProj[16] = {};
    uint32_t serial = 0;  // +1 per focus group (the patched COMMON_BUFFER2 is made again)
};
PrevFocus g_pf;
bool g_velBound = false;  // render thread: render target 0 is a velocity buffer (R16G16_FLOAT)
std::atomic<uint64_t> g_objFixed{0};
int g_dumpCount = 0;
int g_dumpVelocity = 0;  // object velocity draws dumped in this group
constexpr const char* kGroupName[4] = {"outer L", "outer R", "focus L", "focus R"};
std::atomic<int> g_group{-1};  // eye group running (render thread), -1 outside them

// The focus views' TAA history while the image moves (docs/dlaa-trees.md). The engine's resolve PS weighs its history
// by the motion in its view's units: w = 1 - sat((|velocity_uv * (1600, 900)| - 16) * 0.1), and pixels without the
// stencil bit 0x02 (the sky among them) get about 0.95 * w (cameraRotationFactor about 0.02 at w = 0). Above 1% of the
// view per frame the TAA keeps the current frame only. Alpha-tested leaves get their soft edge (the sky pixels next to
// them) from the jittered frames, so they thinned out while the history dropped. A focus view spans about a third of
// the eye's angle and follows the gaze, so it passed the threshold at a third of outer's head speed and with every eye
// movement. The focus groups draw the resolve with a copy of the shader whose literal (1600, 900, 160, 90) is 0: the
// weight stays as at rest (the neighbourhood clip of the history stays, as for the bit 0x02 pixels' fixed 0.75).
constexpr float kTaaMotion[4] = {1600.0f, 900.0f, 160.0f, 90.0f};
struct ResolveCode {
    void* ps = nullptr;         // the engine's resolve PS (compared only)
    std::vector<uint8_t> code;  // its bytecode
    ID3D11PixelShader* patched = nullptr;
    bool failed = false;
};
std::mutex g_resolveM;
ResolveCode g_resolveCode[4];  // under g_resolveM (normally one)
std::atomic<bool> g_motionFix{true};
std::atomic<uint64_t> g_motionFixed{0};
ID3D11PixelShader* g_psRestore = nullptr;  // render thread: the engine's resolve PS, bound again after the draw

void SetState(const char* s) { snprintf(g_state, sizeof(g_state), "%s", s); }

void Fail(const char* where, unsigned long code) {
    if (g_dead.exchange(true)) return;
    Log("TAA views: ERROR: exception 0x%08lX in %s; off for this session (the focus views share outer R's TAA "
        "state)", code, where);
}

template <class F>
void Protect(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Fail(where, code);
    }
}

uint8_t* CurrentTaa() {
    void* dev = nullptr;
    if (!g_devGlobal || !hooks::TryRead(g_devGlobal, &dev, sizeof(dev)) || !dev) return nullptr;
    uint8_t* taa = nullptr;
    if (!hooks::TryRead((uint8_t*)dev + kDevTaa, &taa, sizeof(taa)) || !taa) return nullptr;
    uintptr_t vt = 0;
    if (!hooks::TryRead(taa, &vt, sizeof(vt)) || vt != hooks::ExeBase() + kTaaVtableRva) return nullptr;
    return taa;
}

// The slot 2 array a helper call fills, or -1.
int SlotArray(void* taa, void** rt, void** tex) {
    uint8_t* t = (uint8_t*)taa;
    for (int k = 0; k < kArrays; ++k)
        if ((uint8_t*)rt == t + kRtArr[k] + kSlot * 8 && (uint8_t*)tex == t + kTexArr[k] + kSlot * 8) return k;
    return -1;
}

void** Field(uint8_t* taa, size_t arr) { return (void**)(taa + arr + kSlot * 8); }
uint8_t* Mat(uint8_t* taa, int m) { return taa + kMatArr[m] + kSlot * kMatSize; }

// Diagnostics: in a shot frame the swaps log the previous projection (matrix 2, a copy of the camera's projection) as
// the group starts and ends: outer R has the eye's, a focus group should end with its focus window's.

void LogProj(const char* what, int eye, const uint8_t* m) {
    const float* f = (const float*)m;
    Log("TAA views: %s %s: previous projection m00 %.5f m11 %.5f m20 %.5f m21 %.5f m22 %.5f m32 %.5f", kEyeName[eye],
        what, f[0], f[5], f[8], f[9], f[10], f[14]);
}

void RestoreLocked() {
    if (!g_sw.on) return;
    if (g_dumpFrame) {
        LogProj("group end (written by its resolve)", g_sw.eye, Mat(g_sw.taa, 2));
    }
    uint8_t* taa = g_sw.taa;
    const int e = g_sw.eye;
    int changed = 0;
    for (int k = 0; k < kArrays; ++k) {
        changed += *Field(taa, kRtArr[k]) != g_c.rt[e][k][0];
        changed += *Field(taa, kTexArr[k]) != g_c.tex[e][k][0];
        *Field(taa, kRtArr[k]) = g_sw.rt[k];
        *Field(taa, kTexArr[k]) = g_sw.tex[k];
    }
    bool resolved = false;
    for (int m = 0; m < kMats; ++m) {
        resolved |= memcmp(Mat(taa, m), g_sw.matIn[m], kMatSize) != 0;
        memcpy(g_c.mat[e][m], Mat(taa, m), kMatSize);
        memcpy(Mat(taa, m), g_sw.mat[m], kMatSize);
    }
    g_c.haveMat[e] = true;
    g_st.resolved[e] += resolved;
    if (changed) {
        ++g_st.changed;
        Log("TAA views: %d slot 2 fields changed by the engine during a %s group; outer R's put back", changed,
            kEyeName[e]);
    }
    g_sw = Swap{};
    g_pf.on = false;
}

void ReleaseCopiesLocked() {
    if (!g_c.taa) return;
    int released = 0;
    for (int e = 0; e < 2; ++e)
        for (int k = 0; k < kArrays; ++k)
            if (g_c.rt[e][k][0] || g_c.tex[e][k][0]) {
                g_releaseArray(nullptr, g_c.rt[e][k], g_c.tex[e][k]);
                ++released;
            }
    Log("TAA views: %d focus target copies released with CTAAControl %p", released, g_c.taa);
    g_c = Copies{};
    SetState("on (no TAA now)");
}

bool __fastcall D_Create(void* taa, uint32_t w, uint32_t h, const char* name, int kind, void** rt, void** tex) {
    const bool r = o_create(taa, w, h, name, kind, rt, tex);
    if (!r || t_inside || g_dead.load(std::memory_order_relaxed)) return r;
    Protect("TAA target helper", [&] {
        const int k = SlotArray(taa, rt, tex);
        if (k < 0) return;
        // TAA off in the game: the copies would only take VRAM (an unknown setting counts as on).
        if (sysinfo::GameSetting("temporalantialiasenabled") == "0") {
            if (!g_offLogged) {
                g_offLogged = true;
                Log("TAA views: TAA is off in the game's settings at start; no focus copies of the TAA targets");
            }
            std::lock_guard lock(g_m);
            SetState("off (TAA off in the game at start)");
            return;
        }
        {
            std::lock_guard lock(g_m);
            if (g_c.taa != taa) {
                if (g_c.taa) {
                    Log("TAA views: new CTAAControl %p while the copies of %p were not released; they stay unused",
                        taa, g_c.taa);
                    g_c = Copies{};
                }
                g_c.taa = taa;
                g_c.w = w;
                g_c.h = h;
            }
        }
        void* nrt[2] = {};
        void* ntex[2] = {};
        bool ok[2] = {};
        t_inside = true;
        for (int e = 0; e < 2; ++e) {
            void* slot3[3] = {};
            void* tex3[3] = {};
            ok[e] = o_create(taa, w, h, kCopyName[e][k], kind, &slot3[0], &tex3[0]);
            nrt[e] = slot3[0];
            ntex[e] = tex3[0];
        }
        t_inside = false;
        std::lock_guard lock(g_m);
        if (g_c.taa != taa) return;  // released meanwhile
        for (int e = 0; e < 2; ++e) {
            g_c.rt[e][k][0] = nrt[e];
            g_c.tex[e][k][0] = ntex[e];
            g_c.made += ok[e] && nrt[e];
            if (!ok[e] || !nrt[e])
                Log("TAA views: the engine could not make the %s copy of %s (%s, %u x %u, kind %d)", kEyeName[e],
                    kArrName[k], name ? name : "?", w, h, kind);
        }
        if (g_c.made == 2 * kArrays) {
            Log("TAA views: CTAAControl %p: focus L and focus R copies of the 6 slot 2 targets made (%u x %u)", taa, w,
                h);
            SetState("on (copies ready)");
        }
    });
    return r;
}

void __fastcall D_ReleaseAll(void* taa) {
    if (!g_dead.load(std::memory_order_relaxed)) {
        Protect("TAA release", [&] {
            std::lock_guard lock(g_m);
            if (g_c.taa != taa) return;
            if (g_sw.on && g_sw.taa == taa) RestoreLocked();
            ReleaseCopiesLocked();
        });
    }
    o_releaseAll(taa);
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    if (!hooks::EngineAvailable()) return false;
    Protect("Install", [&] {
        const uint8_t* load = (const uint8_t*)hooks::Address(Anchor::TaaResolveDeviceLoad);
        if (!load || load[0] != 0x48 || load[1] != 0x8B || load[2] != 0x0D) {
            Log("TAA views: device load does not match; the focus views share outer R's TAA state");
            SetState("off: device load does not match");
            return;
        }
        g_devGlobal = (void**)(load + 7 + *(const int32_t*)(load + 3));
        g_releaseArray = (ReleaseArrayFn)hooks::Address(Anchor::TaaReleaseArray);
        bool ok = g_releaseArray != nullptr;
        ok = ok && hooks::Create(Anchor::TaaCreateTarget, (void*)D_Create, (void**)&o_create);
        ok = ok && hooks::Create(Anchor::TaaReleaseTargets, (void*)D_ReleaseAll, (void**)&o_releaseAll);
        g_installed = ok;
        Log("TAA views: engine hooks %s (CTAAControl target helper and release)",
            ok ? "installed" : "INCOMPLETE; the focus views share outer R's TAA state");
        SetState(ok ? "on (no TAA yet)" : "off: engine hooks failed");
    });
    return g_installed;
}

void OnFrameStart(bool enabled, bool dump) {
    if (!g_installed || g_dead.load(std::memory_order_relaxed)) return;
    g_dumpFrame = dump;
    g_dumping = -1;
    g_fixOn = enabled;
    g_group = -1;
    Protect("frame", [&] {
        std::lock_guard lock(g_m);
        if (g_sw.on) {
            ++g_st.late;
            RestoreLocked();
        }
        g_enabled = enabled;
    });
}

void BeginGroup(int group) {
    if (!g_installed || g_dead.load(std::memory_order_relaxed)) return;
    g_group = group;
    if (g_dumpFrame && (group == 1 || group == 2)) {
        g_dumpCount = 0;
        g_dumpVelocity = 0;
        g_dumping = group;
        Log("TAA views dump: %s group, own TAA state %s; full-screen draws (viewport, render target 0, PS views t0-t7 "
            "as W x H fN, xS = samples):", kGroupName[group], g_enabled ? "on" : "off");
    }
    if (!g_enabled || (group != 2 && group != 3)) return;
    Protect("BeginGroup", [&] {
        std::lock_guard lock(g_m);
        if (g_sw.on) {
            ++g_st.late;
            RestoreLocked();
        }
        uint8_t* taa = CurrentTaa();
        if (!taa) return;  // no TAA (MSAA)
        const int e = group - 2;
        void* rt[kArrays];
        void* tex[kArrays];
        for (int k = 0; k < kArrays; ++k) {
            rt[k] = *Field(taa, kRtArr[k]);
            tex[k] = *Field(taa, kTexArr[k]);
            if (!rt[k] || !tex[k]) return;  // no slot 2 state
        }
        if (taa != g_c.taa || g_c.made != 2 * kArrays) {
            ++g_st.noCopies;
            if (!g_beforeLogged) {
                g_beforeLogged = true;
                Log("TAA views: CTAAControl %p has no focus copies (%s); the focus views share outer R's TAA state "
                    "until the game makes its TAA again (track load)",
                    taa, taa != g_c.taa ? "made before the hooks or with TAA off at start" : "the engine refused some");
            }
            return;
        }
        Swap& s = g_sw;
        s.on = true;
        s.taa = taa;
        s.eye = e;
        for (int k = 0; k < kArrays; ++k) {
            s.rt[k] = rt[k];
            s.tex[k] = tex[k];
            *Field(taa, kRtArr[k]) = g_c.rt[e][k][0];
            *Field(taa, kTexArr[k]) = g_c.tex[e][k][0];
        }
        for (int m = 0; m < kMats; ++m) {
            memcpy(s.mat[m], Mat(taa, m), kMatSize);
            // The first time: outer R's matrices (the copies hold no history yet).
            if (g_c.haveMat[e]) memcpy(Mat(taa, m), g_c.mat[e][m], kMatSize);
            memcpy(s.matIn[m], Mat(taa, m), kMatSize);
        }
        if (g_c.haveMat[e]) {
            g_pf.on = true;
            memcpy(g_pf.viewProj, g_c.mat[e][0], sizeof(g_pf.viewProj));
            memcpy(g_pf.proj, g_c.mat[e][2], sizeof(g_pf.proj));
            ++g_pf.serial;
        }
        if (g_dumpFrame) {
            LogProj("group start, outer R's slot 2", e, s.mat[2]);
            LogProj(g_c.haveMat[e] ? "group start, its own from the last frame" : "group start, none of its own yet", e,
                    Mat(taa, 2));
        }
        ++g_st.swaps[e];
    });
}

void EndGroup() {
    if (!g_installed || g_dead.load(std::memory_order_relaxed)) return;
    g_dumping = -1;
    g_group = -1;
    Protect("EndGroup", [&] {
        std::lock_guard lock(g_m);
        RestoreLocked();
    });
}

namespace {

void ViewDesc(ID3D11View* v, char* out, size_t n, DXGI_FORMAT viewFmt) {
    if (!v) {
        snprintf(out, n, "-");
        return;
    }
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    ID3D11Texture2D* t = nullptr;
    if (r && SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        snprintf(out, n, "%ux%u f%d%s%s", d.Width, d.Height, (int)viewFmt, d.SampleDesc.Count > 1 ? " x" : "",
                 d.SampleDesc.Count > 1 ? std::to_string(d.SampleDesc.Count).c_str() : "");
        t->Release();
    } else {
        snprintf(out, n, "(not 2D)");
    }
    if (r) r->Release();
}

// Constant buffers of the pixel shader (slots 0-4) or of the vertex shader (one slot) as floats: up to 64 constants per
// slot, copied through a staging buffer (the read waits for the GPU; shot frames only).
void DumpConstants(ID3D11DeviceContext* c, const char* what, bool vertex = false, UINT vsSlot = 0) {
    ID3D11DeviceContext1* c1 = nullptr;
    if (FAILED(c->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&c1)) || !c1) return;
    ID3D11Buffer* cb[5] = {};
    UINT first[5] = {}, num[5] = {};
    if (vertex)
        c1->VSGetConstantBuffers1(vsSlot, 1, cb, first, num);
    else
        c1->PSGetConstantBuffers1(0, 5, cb, first, num);
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    D3D11_BUFFER_DESC sd{};
    sd.ByteWidth = 64 * 16;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* staging = nullptr;
    if (dev) dev->CreateBuffer(&sd, nullptr, &staging);
    for (int i = 0; i < 5; ++i) {
        if (!cb[i] || !staging) continue;
        D3D11_BUFFER_DESC bd{};
        cb[i]->GetDesc(&bd);
        const UINT start = first[i] * 16;
        if (start >= bd.ByteWidth) continue;
        UINT bytes = bd.ByteWidth - start;
        if (num[i] && num[i] * 16 < bytes) bytes = num[i] * 16;
        if (bytes > 64 * 16) bytes = 64 * 16;
        D3D11_BOX box{start, 0, 0, start + bytes, 1, 1};
        c->CopySubresourceRegion(staging, 0, 0, 0, 0, cb[i], 0, &box);
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(c->Map(staging, 0, D3D11_MAP_READ, 0, &m))) continue;
        const float* f = (const float*)m.pData;
        const int n = (int)(bytes / 4);
        Log("    %s %s cb%u (buffer %u B, first constant %u, %u constants): %d floats", what, vertex ? "VS" : "PS",
            vertex ? vsSlot : (UINT)i, bd.ByteWidth, first[i], num[i], n);
        for (int j = 0; j < n; j += 16) {
            char line[512];
            int o = 0;
            for (int k = j; k < j + 16 && k < n; ++k)
                o += snprintf(line + o, sizeof(line) - o, "%s%.6g", k == j ? "" : " ", f[k]);
            Log("      [%3d] %s", j, line);
        }
        c->Unmap(staging, 0);
    }
    if (staging) staging->Release();
    if (dev) dev->Release();
    for (ID3D11Buffer* b : cb)
        if (b) b->Release();
    c1->Release();
}

void DumpDraw(ID3D11DeviceContext* c, bool indexed, UINT count) {
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(1, &rtv, &dsv);
    ID3D11ShaderResourceView* srv[8] = {};
    c->PSGetShaderResources(0, 8, srv);
    ID3D11PixelShader* ps = nullptr;
    c->PSGetShader(&ps, nullptr, nullptr);
    char rt[48], ds[48], t[8][48];
    DXGI_FORMAT tf[8] = {};
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    if (rtv) rtv->GetDesc(&rd);
    ViewDesc(rtv, rt, sizeof(rt), rd.Format);
    D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
    if (dsv) dsv->GetDesc(&dd);
    ViewDesc(dsv, ds, sizeof(ds), dd.Format);
    for (int i = 0; i < 8; ++i) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
        if (srv[i]) srv[i]->GetDesc(&sd);
        tf[i] = srv[i] ? sd.Format : DXGI_FORMAT_UNKNOWN;
        ViewDesc(srv[i], t[i], sizeof(t[i]), sd.Format);
    }
    Log("  #%d %s(%u) ps %p vp %.0fx%.0f at (%.0f,%.0f) rt %s ds %s | %s | %s | %s | %s | %s | %s | %s | %s", g_dumpCount,
        indexed ? "DrawIndexed" : "Draw", count, (void*)ps, vp.Width, vp.Height, vp.TopLeftX, vp.TopLeftY, rt, ds, t[0],
        t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    // The TAA passes (formats as in docs/ams2-renderer.md): resolve, locks and the camera velocity pass.
    const DXGI_FORMAT rf = rtv ? rd.Format : DXGI_FORMAT_UNKNOWN;
    const char* what = nullptr;
    if (rf == DXGI_FORMAT_R11G11B10_FLOAT && tf[3] == DXGI_FORMAT_R16G16_FLOAT && tf[5] == DXGI_FORMAT_R8_UNORM)
        what = "TAA resolve";
    else if (rf == DXGI_FORMAT_R8_UNORM && tf[3] == DXGI_FORMAT_R16G16_FLOAT)
        what = "TAA locks";
    else if (rf == DXGI_FORMAT_R16G16_FLOAT && tf[0] == DXGI_FORMAT_R32_FLOAT && !srv[1])
        what = "camera velocity";
    if (what) DumpConstants(c, what);
    if (ps) ps->Release();
    if (rtv) rtv->Release();
    if (dsv) dsv->Release();
    for (ID3D11ShaderResourceView* s : srv)
        if (s) s->Release();
}

}  // namespace

namespace {

// viewportScale fix. The TAA resolve and locks shaders read their pixels at uv * viewportScale.zw (cb0 of the PS:
// 1/w, 1/h, w, h, then the accumulation rates). The engine keeps that cb0 at one fixed place of its constant buffer
// for every view, so the GPU sees the size of the view written last: with clones (views in different sizes) outer
// read the focus size and showed a 1.333x zoomed image (dlaa-plan D2, test 4). Before each such draw the PS gets this
// module's cb0: the engine's 32 bytes copied on the GPU into a plain buffer, the first 16 replaced there by the size of
// the bound render target, then the whole buffer copied into the constant buffer (constant buffers take no partial
// updates without NO_OVERWRITE or DISCARD, and this one is reused from draw to draw).
// COMMON_BUFFER2 of the engine's shaders (docs/ams2-renderer.md): 976 bytes, prevProjMatrix at +816 and prevViewProj
// right after it at +880.
constexpr UINT kCommon2Size = 976;
constexpr UINT kPrevProjOffset = 816;

struct Fix {
    ID3D11Device* dev = nullptr;  // not referenced (compared only)
    ID3D11DeviceContext1* c1 = nullptr;
    ID3D11Buffer* cb = nullptr;
    ID3D11Buffer* scratch = nullptr;  // plain buffer for the copy and the partial update
    bool failed = false;
    // Bound before the draw, put back after it.
    bool restore = false;
    ID3D11Buffer* buf = nullptr;
    UINT first = 0, num = 0;
    // Object motion vectors: this module's COMMON_BUFFER2 for the vertex shader (slot 12).
    ID3D11Buffer* cb12 = nullptr;
    ID3D11Buffer* scratch12 = nullptr;
    bool restore12 = false;
    ID3D11Buffer* buf12 = nullptr;
    UINT first12 = 0, num12 = 0;
    const void* madeFrom = nullptr;  // engine buffer, first constant and group the cb12 was made from (compared only)
    UINT madeFirst = 0;
    uint32_t madeSerial = 0;
};
Fix g_fx;  // render thread
ID3D11DeviceContext* g_testCtx = nullptr;  // smoke test: the context in place of the game's

ID3D11DeviceContext* GameContext() { return g_testCtx ? g_testCtx : gpu::Context(); }
std::atomic<uint64_t> g_fixed[2];  // resolve, locks
uint32_t g_fixLogged[4][3] = {};   // (kind + 1, w, h) logged once

void DropFix() {
    if (g_fx.c1) g_fx.c1->Release();
    if (g_fx.cb) g_fx.cb->Release();
    if (g_fx.scratch) g_fx.scratch->Release();
    if (g_fx.buf) g_fx.buf->Release();
    if (g_fx.cb12) g_fx.cb12->Release();
    if (g_fx.scratch12) g_fx.scratch12->Release();
    if (g_fx.buf12) g_fx.buf12->Release();
    g_fx = Fix{};
}

bool EnsureFix(ID3D11DeviceContext* c) {
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return false;
    dev->Release();  // the context keeps it
    if (dev == g_fx.dev) return !g_fx.failed;
    DropFix();
    g_fx.dev = dev;
    D3D11_FEATURE_DATA_D3D11_OPTIONS o{};
    if (FAILED(dev->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &o, sizeof(o))) || !o.ConstantBufferOffsetting) {
        Log("TAA views: the device has no constant buffer offsets; the TAA viewportScale fix is off");
        g_fx.failed = true;
        return false;
    }
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = 32;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_BUFFER_DESC sd = d;
    sd.BindFlags = 0;
    if (FAILED(c->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_fx.c1)) ||
        FAILED(dev->CreateBuffer(&d, nullptr, &g_fx.cb)) || FAILED(dev->CreateBuffer(&sd, nullptr, &g_fx.scratch))) {
        Log("TAA views: cannot make the constant buffer of the TAA viewportScale fix; the fix is off");
        g_fx.failed = true;
        return false;
    }
    d.ByteWidth = sd.ByteWidth = kCommon2Size;
    if (FAILED(dev->CreateBuffer(&d, nullptr, &g_fx.cb12)) || FAILED(dev->CreateBuffer(&sd, nullptr, &g_fx.scratch12))) {
        Log("TAA views: cannot make the COMMON_BUFFER2 copy of the object velocity fix; the TAA fixes are off");
        g_fx.failed = true;
        return false;
    }
    return true;
}

// 0 resolve, 1 locks, -1 another draw. Cheap for the other draws: render target 0's format first.
int TaaKind(ID3D11DeviceContext* c, UINT& w, UINT& h) {
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    if (!rtv) return -1;
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    rtv->GetDesc(&rd);
    int kind = -1;
    if (rd.Format == DXGI_FORMAT_R11G11B10_FLOAT || rd.Format == DXGI_FORMAT_R8_UNORM) {
        ID3D11ShaderResourceView* s[6] = {};
        c->PSGetShaderResources(0, 6, s);
        DXGI_FORMAT f[6] = {};
        for (int i = 0; i < 6; ++i) {
            if (!s[i]) continue;
            D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
            s[i]->GetDesc(&sd);
            f[i] = sd.Format;
            s[i]->Release();
        }
        if (f[3] == DXGI_FORMAT_R16G16_FLOAT) {
            if (rd.Format == DXGI_FORMAT_R11G11B10_FLOAT && f[5] == DXGI_FORMAT_R8_UNORM) kind = 0;
            if (rd.Format == DXGI_FORMAT_R8_UNORM) kind = 1;
        }
        if (kind >= 0) {
            ID3D11Resource* r = nullptr;
            rtv->GetResource(&r);
            ID3D11Texture2D* t = nullptr;
            if (r && SUCCEEDED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t)) && t) {
                D3D11_TEXTURE2D_DESC td{};
                t->GetDesc(&td);
                w = td.Width;
                h = td.Height;
                t->Release();
            } else {
                kind = -1;
            }
            if (r) r->Release();
        }
    }
    rtv->Release();
    return kind;
}

// In a focus group the TAA resolve draws with its copy that keeps the history while the image moves (kTaaMotion). The
// copy is made on first use per engine shader.
void SwapResolveShader(ID3D11DeviceContext* c) {
    const int g = g_group.load(std::memory_order_relaxed);
    if (!g_motionFix.load(std::memory_order_relaxed) || (g != 2 && g != 3) || g_psRestore) return;
    ID3D11PixelShader* ps = nullptr;
    c->PSGetShader(&ps, nullptr, nullptr);
    if (!ps) return;
    ID3D11PixelShader* patched = nullptr;
    {
        std::lock_guard lock(g_resolveM);
        for (auto& r : g_resolveCode) {
            if (r.ps != ps) continue;
            if (!r.patched && !r.failed) {
                r.failed = true;
                const float zero[4] = {};
                std::vector<uint8_t> out;
                const int n = dxbc::ReplaceImmediate4(r.code.data(), r.code.size(), kTaaMotion, zero, out);
                ID3D11Device* dev = nullptr;
                c->GetDevice(&dev);
                if (n == 1 && dev && SUCCEEDED(dev->CreatePixelShader(out.data(), out.size(), nullptr, &r.patched)))
                    r.failed = false;
                if (dev) dev->Release();
                Log("TAA views: focus TAA resolve that keeps its history while the image moves: %s",
                    !r.failed ? "made" : n == 1 ? "shader not made" : "motion constant not found once");
            }
            patched = r.patched;
            break;
        }
    }
    if (!patched) {
        ps->Release();
        return;  // not the known resolve shader, or its copy failed
    }
    c->PSSetShader(patched, nullptr, 0);
    g_psRestore = ps;  // keeps the reference until AfterDraw
    g_motionFixed.fetch_add(1, std::memory_order_relaxed);
}

void FixDraw(ID3D11DeviceContext* c) {
    UINT w = 0, h = 0;
    const int kind = TaaKind(c, w, h);
    if (kind < 0 || !w || !h || !EnsureFix(c)) return;
    ID3D11Buffer* eb = nullptr;
    UINT first = 0, num = 0;
    g_fx.c1->PSGetConstantBuffers1(0, 1, &eb, &first, &num);
    if (!eb) return;
    D3D11_BUFFER_DESC bd{};
    eb->GetDesc(&bd);
    const UINT start = first * 16;
    if (eb == g_fx.cb || start + 32 > bd.ByteWidth) {
        eb->Release();
        return;
    }
    const D3D11_BOX src{start, 0, 0, start + 32, 1, 1};
    c->CopySubresourceRegion(g_fx.scratch, 0, 0, 0, 0, eb, 0, &src);
    const float v[4] = {1.0f / w, 1.0f / h, (float)w, (float)h};
    const D3D11_BOX part{0, 0, 0, 16, 1, 1};
    c->UpdateSubresource(g_fx.scratch, 0, &part, v, 0, 0);
    c->CopyResource(g_fx.cb, g_fx.scratch);
    g_fx.restore = true;
    g_fx.buf = eb;  // keeps the reference until AfterDraw
    g_fx.first = first;
    g_fx.num = num;
    c->PSSetConstantBuffers(0, 1, &g_fx.cb);
    g_fixed[kind].fetch_add(1, std::memory_order_relaxed);
    if (kind == 0) SwapResolveShader(c);
    for (auto& l : g_fixLogged) {
        if (l[0] == (uint32_t)kind + 1 && l[1] == w && l[2] == h) break;
        if (l[0]) continue;
        l[0] = kind + 1;
        l[1] = w;
        l[2] = h;
        Log("TAA views: TAA %s at %u x %u gets its own viewportScale (the engine's cb0 is shared by all views)",
            kind ? "locks" : "resolve", w, h);
        break;
    }
}

// Object motion vectors in the focus groups. The velocity pass of objects (cars, wheels, driver, grass) projects the
// previous position with COMMON_BUFFER2's prevViewProj, which the engine builds from the previous projection of the
// view 2 camera: in a focus group that is the focus window of this frame, not of the last one. While the window
// follows the gaze, moving objects then got motion vectors without the window's own motion and smeared (dlaa-plan D2,
// test 8). In a focus group each draw into the velocity buffer gets a copy of the engine's COMMON_BUFFER2 with
// prevProjMatrix and prevViewProj replaced by this focus view's of the last frame (the same matrices the camera
// velocity pass gets from the TAA state). The copy is made once per engine buffer place and group.
void FixObjectVelocity(ID3D11DeviceContext* c) {
    if (!EnsureFix(c)) return;
    ID3D11Buffer* eb = nullptr;
    UINT first = 0, num = 0;
    g_fx.c1->VSGetConstantBuffers1(12, 1, &eb, &first, &num);
    if (!eb) return;
    D3D11_BUFFER_DESC bd{};
    eb->GetDesc(&bd);
    const UINT start = first * 16;
    if (eb == g_fx.cb12 || start + kCommon2Size > bd.ByteWidth) {
        eb->Release();
        return;
    }
    if (eb != g_fx.madeFrom || first != g_fx.madeFirst || g_pf.serial != g_fx.madeSerial) {
        const D3D11_BOX src{start, 0, 0, start + kCommon2Size, 1, 1};
        c->CopySubresourceRegion(g_fx.scratch12, 0, 0, 0, 0, eb, 0, &src);
        float m[32];
        memcpy(m, g_pf.proj, sizeof(g_pf.proj));
        memcpy(m + 16, g_pf.viewProj, sizeof(g_pf.viewProj));
        const D3D11_BOX part{kPrevProjOffset, 0, 0, kPrevProjOffset + sizeof(m), 1, 1};
        c->UpdateSubresource(g_fx.scratch12, 0, &part, m, 0, 0);
        c->CopyResource(g_fx.cb12, g_fx.scratch12);
        g_fx.madeFrom = eb;
        g_fx.madeFirst = first;
        g_fx.madeSerial = g_pf.serial;
    }
    g_fx.restore12 = true;
    g_fx.buf12 = eb;  // keeps the reference until AfterDraw
    g_fx.first12 = first;
    g_fx.num12 = num;
    c->VSSetConstantBuffers(12, 1, &g_fx.cb12);
    g_objFixed.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

// Diagnostics: the first two object draws into the velocity buffer (R16G16_FLOAT) of a dumped group log the vertex
// shader's COMMON_BUFFER2 (slot 12: viewProj at constant 12, prevView 47, prevProjMatrix 51, prevViewProj 55,
// cameraJitter 59).

void DumpObjectVelocity(ID3D11DeviceContext* c, UINT count) {
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    if (!rtv) return;
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    rtv->GetDesc(&rd);
    rtv->Release();
    if (rd.Format != DXGI_FORMAT_R16G16_FLOAT || g_dumpVelocity >= 2) return;
    ++g_dumpVelocity;
    Log("  object velocity draw (%u indices), %s:", count, kGroupName[g_dumping.load()]);
    DumpConstants(c, "object velocity", true, 12);
}

bool OnDraw(ID3D11DeviceContext* c, bool indexed, UINT count) {
    const bool dumping = g_dumping.load(std::memory_order_relaxed) >= 0;
    const bool fixing = g_fixOn.load(std::memory_order_relaxed);
    if (!dumping && !fixing) return false;
    if (c != GameContext()) return false;
    if (count > 6) {
        if (dumping) Protect("dump", [&] { DumpObjectVelocity(c, count); });
    } else {
        if (dumping && g_dumpCount < kDumpDraws) {
            ++g_dumpCount;
            Protect("dump", [&] { DumpDraw(c, indexed, count); });
        }
        if (fixing && indexed && count == 6) Protect("viewportScale fix", [&] { FixDraw(c); });
    }
    if (fixing && g_velBound && g_pf.on) Protect("object velocity fix", [&] { FixObjectVelocity(c); });
    return g_fx.restore || g_fx.restore12 || g_psRestore;
}

void AfterDraw(ID3D11DeviceContext* c) {
    (void)c;
    if (g_psRestore)
        Protect("resolve shader restore", [&] {
            g_fx.c1->PSSetShader(g_psRestore, nullptr, 0);
            g_psRestore->Release();
            g_psRestore = nullptr;
        });
    if (g_fx.restore)
        Protect("viewportScale restore", [&] {
            g_fx.c1->PSSetConstantBuffers1(0, 1, &g_fx.buf, &g_fx.first, &g_fx.num);
            g_fx.buf->Release();
            g_fx.buf = nullptr;
            g_fx.restore = false;
        });
    if (g_fx.restore12)
        Protect("object velocity restore", [&] {
            g_fx.c1->VSSetConstantBuffers1(12, 1, &g_fx.buf12, &g_fx.first12, &g_fx.num12);
            g_fx.buf12->Release();
            g_fx.buf12 = nullptr;
            g_fx.restore12 = false;
        });
}

void OnSetTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv) {
    if (c != GameContext()) return;
    bool vel = false;
    if (n && rtv && rtv[0]) {
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        rtv[0]->GetDesc(&d);
        vel = d.Format == DXGI_FORMAT_R16G16_FLOAT;
    }
    g_velBound = vel;
}

void TestSetPrevFocus(const float proj[16], const float viewProj[16]) {
    g_pf.on = proj != nullptr;
    if (proj) {
        memcpy(g_pf.proj, proj, sizeof(g_pf.proj));
        memcpy(g_pf.viewProj, viewProj, sizeof(g_pf.viewProj));
        ++g_pf.serial;
    }
}

uint64_t TestObjectDraws() { return g_objFixed.load(); }

void OnPixelShaderMade(ID3D11PixelShader* ps, const void* code, size_t len) {
    if (!ps || !code) return;
    // The TAA resolve: its motion constant (as an immediate operand) and the name of cameraRotationFactor.
    const uint8_t* b = (const uint8_t*)code;
    uint8_t pat[20];
    const uint32_t tok = 0x00004002u;
    memcpy(pat, &tok, 4);
    memcpy(pat + 4, kTaaMotion, 16);
    static const char kName[] = "cameraRotationFactor";
    const bool resolve = std::search(b, b + len, pat, pat + 20) != b + len &&
                         std::search(b, b + len, kName, kName + sizeof(kName) - 1) != b + len;
    std::lock_guard lock(g_resolveM);
    for (auto& r : g_resolveCode)
        if (r.ps == ps) {  // a new shader at the address of a released one
            if (r.patched) r.patched->Release();
            r = ResolveCode{};
        }
    if (!resolve) return;
    ResolveCode* slot = nullptr;
    for (auto& r : g_resolveCode)
        if (!r.ps) {
            slot = &r;
            break;
        }
    if (!slot) {
        Log("TAA views: a fifth TAA resolve pixel shader (%p): the focus views keep the game's history weight there",
            (void*)ps);
        return;
    }
    slot->ps = ps;
    slot->code.assign(b, b + len);
    Log("TAA views: the game made its TAA resolve pixel shader (%p, %zu bytes)", (void*)ps, len);
}

void SetMotionFix(bool on) { g_motionFix = on; }
bool GetMotionFix() { return g_motionFix.load(); }
void TestSetGroup(int group) { g_group = group; }
uint64_t TestMotionFixed() { return g_motionFixed.load(); }

void TestEnableFix(ID3D11DeviceContext* c, bool on) {
    g_testCtx = c;
    g_fixOn = on;
}

uint64_t TestFixedDraws() { return g_fixed[0].load() + g_fixed[1].load(); }

std::string Describe() {
    if (!g_installed) return std::string("TAA views ") + g_state;
    if (g_dead.load()) return "TAA views stopped after an exception";
    std::lock_guard lock(g_m);
    return std::string("TAA views ") + g_state + (g_enabled ? "" : ", now off (setting or hotkey)");
}

void LogStats(bool totals) {
    if (!g_installed || g_dead.load()) return;
    std::lock_guard lock(g_m);
    for (int e = 0; e < 2; ++e) {
        g_tot.swaps[e] += g_st.swaps[e];
        g_tot.resolved[e] += g_st.resolved[e];
    }
    g_tot.noCopies += g_st.noCopies;
    g_tot.changed += g_st.changed;
    g_tot.late += g_st.late;
    const Stats& s = totals ? g_tot : g_st;
    Log("TAA views%s: viewportScale fix: %llu resolve and %llu locks draws so far; focus object velocity draws with "
        "the focus view's previous matrices %llu; focus resolves keeping the history while the image moves %llu (%s)",
        totals ? " (session total)" : "", (unsigned long long)g_fixed[0].load(), (unsigned long long)g_fixed[1].load(),
        (unsigned long long)g_objFixed.load(), (unsigned long long)g_motionFixed.load(),
        g_motionFix.load() ? "on" : "off");
    if (s.swaps[0] || s.swaps[1] || s.noCopies || s.changed || s.late)
        Log("TAA views stats%s: own TAA state in focus L/R groups %llu/%llu (slot 2 matrices updated in %llu/%llu), "
            "groups without copies %llu, fields changed by the engine %llu, late restores %llu",
            totals ? " (session total)" : "", (unsigned long long)s.swaps[0], (unsigned long long)s.swaps[1],
            (unsigned long long)s.resolved[0], (unsigned long long)s.resolved[1], (unsigned long long)s.noCopies,
            (unsigned long long)s.changed, (unsigned long long)s.late);
    g_st = Stats{};
}

void Shutdown() {
    if (!g_installed || g_dead.load()) return;
    Protect("Shutdown", [&] {
        std::lock_guard lock(g_m);
        RestoreLocked();
        g_enabled = false;
    });
    g_fixOn = false;
}

}  // namespace qvfr::taaviews
