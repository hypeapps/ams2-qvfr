#include "dlaa.h"

#include <d3d11_1.h>

#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "common.h"
#include "diag.h"
#include "gpu.h"
#include "hooks.h"
#include "sysinfo.h"

#ifdef QVFR_HAVE_DLSS
#include <nvsdk_ngx_helpers_d3d.h>
#endif

namespace qvfr::dlaa {

namespace {

using hooks::Anchor;

// CTAAControl resolve and locks (vtable slots 8 and 7): this, source color, normal/stencil, render context, camera,
// an 8-byte argument, view slot. They record into the engine's command buffers; the D3D11 draw comes at replay.
using TaaFn = void(__fastcall*)(void* taa, void* src, void* normal, uint32_t ctx, void* camera, uint64_t a6, int view);
using DrawIndexedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, INT);

TaaFn o_resolve = nullptr;
TaaFn o_locks = nullptr;
DrawIndexedFn o_drawIndexed = nullptr;

constexpr size_t kJitterOffset = 0x1e50;  // engine device: float2 projection offset per render context (SetJitter)
constexpr uint32_t kMaxCtx = 16;
constexpr int kMaxEvalFails = 20;  // consecutive failed evaluations before the game's TAA takes over for the session
enum { kNotTaa, kDone, kFailed };  // what a DrawIndexed(6) turned out to be

bool g_installed = false;
bool g_failed = false;               // a hard failure (NGX, faults): the game's TAA for the rest of the session
std::atomic<bool> g_on{false};       // hooks in place and not failed
std::atomic<bool> g_active{true};    // hotkey: DLAA (true) or the game's TAA
void** g_devGlobal = nullptr;        // address of the engine device pointer
bool g_d3dHooked = false;
bool g_keyDown = false;
std::atomic<uint64_t> g_frame{0};
std::atomic<uint64_t> g_lastOkFrame{0};  // frame of the last DLSS evaluation (locks may be skipped)
std::atomic<uint32_t> g_lastCtx{0};      // render context of the last recorded TAA resolve (jitter index)
int g_diagLogs = 0;

// SetJitter (device vtable +0x940): the last non-zero offset and the frame counter it was set at (render thread).
using SetJitterFn = void(__fastcall*)(void* dev, const float* offset, uint32_t ctx);
SetJitterFn o_setJitter = nullptr;
constexpr size_t kFrameCounterOffset = 0x1dcc;  // engine device: frame counter (device vtable +0x930)
float g_jit[2] = {0, 0};
uint32_t g_jitCounter = 0xFFFFFFFFu;
uint32_t g_jitCtx = 0;
uint64_t g_setCalls = 0, g_setNonzero = 0, g_lastSetCalls = 0, g_lastSetNonzero = 0;
int g_setLogs = 0;
bool g_firstEvalLogged = false;
bool g_faultLogged = false;
thread_local bool t_inside = false;  // inside our evaluation

std::atomic<uint64_t> g_evals{0}, g_fallbacks{0}, g_locksSkipped{0}, g_resolves{0};
uint64_t g_lastEvals = 0, g_lastFallbacks = 0, g_lastLocks = 0, g_lastResolves = 0;
std::mutex g_sm;
std::string g_state = "not started";

void SetState(const std::string& s) {
    std::lock_guard lock(g_sm);
    g_state = s;
    diag::Set("DLAA", s);
}

template <class F>
bool Protect(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
        return true;
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (!g_faultLogged) {
            g_faultLogged = true;
            Log("ERROR: exception 0x%08lX in DLAA (%s); the game's TAA for the rest of the session", code, where);
        }
        g_failed = true;
        g_on = false;
        return false;
    }
}

bool Pressed(int mod, int key, bool& wasDown) {
    bool down = key && (GetAsyncKeyState(key) & 0x8000) && (!mod || (GetAsyncKeyState(mod) & 0x8000));
    bool pressed = down && !wasDown;
    wasDown = down;
    return pressed;
}

template <class T>
void SafeRelease(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

bool IsTypeless(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R32G32_TYPELESS:
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R16G16_TYPELESS:
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_R8G8_TYPELESS:
        case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_R8_TYPELESS:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        case DXGI_FORMAT_B8G8R8X8_TYPELESS:
            return true;
        default:
            return false;
    }
}

#ifdef QVFR_HAVE_DLSS

// ---------------------------------------------------------------- NGX and DLSS

constexpr const char* kProjectId = "6f1d3a8e-2c4b-4e7a-9b15-a2e5c9f40001";  // NGX project id (GUID form, ours)

struct Feature {
    NVSDK_NGX_Handle* handle = nullptr;
    uintptr_t key = 0;  // the motion vector texture
    UINT w = 0, h = 0;
    int flags = 0;
    int preset = -1;
    uint64_t lastFrame = 0;
    bool fresh = true;
};

struct Scratch {
    ID3D11Texture2D* tex = nullptr;
    UINT w = 0, h = 0, mips = 0, arr = 0;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    UINT bind = 0;
};

ID3D11Device* g_dev = nullptr;  // referenced; NGX lives on this device
ID3D11DeviceContext1* g_ctx1 = nullptr;
ID3DDeviceContextState* g_ctxState = nullptr;
uint32_t g_gen = 0;
bool g_ngxInit = false;
NVSDK_NGX_Parameter* g_params = nullptr;
std::vector<Feature> g_feat;
std::vector<Scratch> g_scratch;
int g_evalFails = 0;
std::atomic<int> g_ngxLogLines{0};
bool g_formatsLogged = false;

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    if (!message || g_ngxLogLines.fetch_add(1) >= 300) return;
    std::string s(message);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    Log("NGX: %s", s.c_str());
}

std::string ResultText(NVSDK_NGX_Result r) {
    char buf[64];
    snprintf(buf, sizeof(buf), "0x%08X", (unsigned)r);
    const wchar_t* w = GetNGXResultAsString(r);
    return w ? Narrow(w) + " (" + buf + ")" : std::string(buf);
}

void DropNgxLocked() {
    for (Feature& f : g_feat)
        if (f.handle) NVSDK_NGX_D3D11_ReleaseFeature(f.handle);
    g_feat.clear();
    for (Scratch& s : g_scratch) SafeRelease(s.tex);
    g_scratch.clear();
    if (g_params) NVSDK_NGX_D3D11_DestroyParameters(g_params);
    g_params = nullptr;
    if (g_ngxInit && g_dev) NVSDK_NGX_D3D11_Shutdown1(g_dev);
    g_ngxInit = false;
    SafeRelease(g_ctxState);
    SafeRelease(g_ctx1);
    SafeRelease(g_dev);
}

bool InitNgx(ID3D11DeviceContext* c) {
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!dev) return false;
    g_dev = dev;
    g_gen = gpu::Generation();
    if (FAILED(c->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&g_ctx1)) || !g_ctx1) {
        Log("DLAA: no ID3D11DeviceContext1; the game's TAA stays");
        return false;
    }
    ID3D11Device1* dev1 = nullptr;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device1), (void**)&dev1)) || !dev1) {
        Log("DLAA: no ID3D11Device1; the game's TAA stays");
        return false;
    }
    D3D_FEATURE_LEVEL fl = dev->GetFeatureLevel();
    HRESULT hr = dev1->CreateDeviceContextState(0, &fl, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device1), nullptr,
                                                &g_ctxState);
    dev1->Release();
    if (FAILED(hr) || !g_ctxState) {
        Log("DLAA: CreateDeviceContextState failed (0x%08lX); the game's TAA stays", (unsigned long)hr);
        return false;
    }
    const std::wstring dllDir = (ModuleDir() / L"qvfr").wstring();
    const wchar_t* paths[] = {dllDir.c_str()};
    NVSDK_NGX_FeatureCommonInfo fi{};
    fi.PathListInfo.Path = paths;
    fi.PathListInfo.Length = 1;
    fi.LoggingInfo.LoggingCallback = NgxLog;
    fi.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    fi.LoggingInfo.DisableOtherLoggingSinks = true;
    const std::wstring data = LogDir().wstring();
    NVSDK_NGX_Result r = NVSDK_NGX_D3D11_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1",
                                                             data.c_str(), dev, &fi, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        Log("DLAA: NGX init failed: %s; the game's TAA stays", ResultText(r).c_str());
        SetState("off: NVIDIA NGX init failed (" + ResultText(r) + ")");
        return false;
    }
    g_ngxInit = true;
    NVSDK_NGX_Parameter* caps = nullptr;
    r = NVSDK_NGX_D3D11_GetCapabilityParameters(&caps);
    int avail = 0, needsDriver = 0, initResult = 0;
    if (NVSDK_NGX_SUCCEED(r) && caps) {
        NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
        NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver);
        NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
        NVSDK_NGX_D3D11_DestroyParameters(caps);
    }
    Log("DLAA: NGX ready on device %p; DLSS available %d, needs a newer driver %d, init result 0x%08X", (void*)dev,
        avail, needsDriver, (unsigned)initResult);
    if (!avail) {
        SetState(needsDriver ? "off: DLSS needs a newer NVIDIA driver" : "off: DLSS not available on this GPU");
        return false;
    }
    r = NVSDK_NGX_D3D11_AllocateParameters(&g_params);
    if (NVSDK_NGX_FAILED(r) || !g_params) {
        Log("DLAA: NGX parameters failed: %s", ResultText(r).c_str());
        return false;
    }
    SetState("on (DLSS ready)");
    return true;
}

// A texture we own that matches `like` (size, mips, array) in format `fmt` with the given bind flags.
ID3D11Texture2D* ScratchFor(const D3D11_TEXTURE2D_DESC& like, DXGI_FORMAT fmt, UINT bind) {
    for (Scratch& s : g_scratch)
        if (s.tex && s.w == like.Width && s.h == like.Height && s.mips == like.MipLevels && s.arr == like.ArraySize &&
            s.fmt == fmt && s.bind == bind)
            return s.tex;
    if (g_scratch.size() >= 8) {
        SafeRelease(g_scratch.front().tex);
        g_scratch.erase(g_scratch.begin());
    }
    D3D11_TEXTURE2D_DESC d{};
    d.Width = like.Width;
    d.Height = like.Height;
    d.MipLevels = like.MipLevels;
    d.ArraySize = like.ArraySize;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = bind;
    Scratch s{};
    HRESULT hr = g_dev->CreateTexture2D(&d, nullptr, &s.tex);
    if (FAILED(hr) || !s.tex) {
        Log("DLAA: texture %ux%u format %d bind 0x%X failed (0x%08lX)", d.Width, d.Height, (int)fmt, bind,
            (unsigned long)hr);
        return nullptr;
    }
    s.w = d.Width;
    s.h = d.Height;
    s.mips = d.MipLevels;
    s.arr = d.ArraySize;
    s.fmt = fmt;
    s.bind = bind;
    g_scratch.push_back(s);
    Log("DLAA: texture %ux%u format %d bind 0x%X created", d.Width, d.Height, (int)fmt, bind);
    return s.tex;
}

// One DLSS feature per motion vector texture (the engine keeps one per view slot) and size.
Feature* FeatureFor(ID3D11DeviceContext* c, uintptr_t key, UINT w, UINT h) {
    const Config& cfg = Cfg();
    int flags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (cfg.dlaaMvJittered) flags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
    if (cfg.dlaaAutoExposure) flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    const int preset = cfg.dlaaPreset;
    Feature* f = nullptr;
    for (Feature& x : g_feat)
        if (x.key == key && x.w == w && x.h == h) f = &x;
    if (f && f->handle && f->flags == flags && f->preset == preset) return f;
    if (!f) {
        if (g_feat.size() >= 8) {
            if (g_feat.front().handle) NVSDK_NGX_D3D11_ReleaseFeature(g_feat.front().handle);
            g_feat.erase(g_feat.begin());
        }
        g_feat.push_back(Feature{});
        f = &g_feat.back();
        f->key = key;
        f->w = w;
        f->h = h;
    }
    if (f->handle) NVSDK_NGX_D3D11_ReleaseFeature(f->handle);
    f->handle = nullptr;
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, (unsigned)preset);
    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth = w;
    cp.Feature.InHeight = h;
    cp.Feature.InTargetWidth = w;
    cp.Feature.InTargetHeight = h;
    cp.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
    cp.InFeatureCreateFlags = flags;
    NVSDK_NGX_Result r = NGX_D3D11_CREATE_DLSS_EXT(c, &f->handle, g_params, &cp);
    if (NVSDK_NGX_FAILED(r) || !f->handle) {
        Log("DLAA: DLSS feature for motion vectors %p (%ux%u, flags 0x%X, preset %d) failed: %s", (void*)key, w, h,
            flags, preset, ResultText(r).c_str());
        f->handle = nullptr;
        return nullptr;
    }
    f->flags = flags;
    f->preset = preset;
    f->fresh = true;
    Log("DLAA: DLSS feature for motion vectors %p created (%ux%u, DLAA, flags 0x%X, preset %d)", (void*)key, w, h, flags,
        preset);
    return f;
}

void ResetAllLocked() {
    for (Feature& f : g_feat) f.fresh = true;
}

DXGI_FORMAT ViewFormat(ID3D11ShaderResourceView* v) {
    if (!v) return DXGI_FORMAT_UNKNOWN;
    D3D11_SHADER_RESOURCE_VIEW_DESC d;
    v->GetDesc(&d);
    return d.Format;
}

// A DrawIndexed(6) on the immediate context. The engine records its TAA resolve into a command buffer and replays it
// later, so the draw is recognized by what is bound (docs/ams2-renderer.md): target and history R11G11B10_FLOAT,
// t3 motion vectors R16G16_FLOAT, t5 locks R8_UNORM. Returns kNotTaa, kDone (DLSS ran, skip the draw) or kFailed.
int Evaluate(ID3D11DeviceContext* c) {
    ID3D11ShaderResourceView* srv[6] = {};
    c->PSGetShaderResources(0, 6, srv);
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    ID3D11Texture2D* tex[4] = {};  // depth, color, motion vectors, target
    int result = kNotTaa;
    [&] {
        if (!rtv || !srv[0] || !srv[1] || !srv[3]) return;
        D3D11_RENDER_TARGET_VIEW_DESC rd;
        rtv->GetDesc(&rd);
        if (rd.Format != DXGI_FORMAT_R11G11B10_FLOAT || ViewFormat(srv[2]) != DXGI_FORMAT_R11G11B10_FLOAT ||
            ViewFormat(srv[3]) != DXGI_FORMAT_R16G16_FLOAT || ViewFormat(srv[5]) != DXGI_FORMAT_R8_UNORM)
            return;
        result = kFailed;  // from here on it is the TAA resolve
        if (g_failed) return;
        if (g_dev && g_gen != gpu::Generation()) {
            Log("DLAA: the device changed; NGX starts again");
            DropNgxLocked();
        }
        if (!g_ngxInit && !g_dev && !InitNgx(c)) {
            g_failed = true;
            g_on = false;
            DropNgxLocked();
            return;
        }
        if (!g_params) return;
        ID3D11View* views[4] = {srv[0], srv[1], srv[3], rtv};
        for (int i = 0; i < 4; ++i) {
            ID3D11Resource* r = nullptr;
            views[i]->GetResource(&r);
            if (!r) return;
            HRESULT hr = r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex[i]);
            r->Release();
            if (FAILED(hr) || !tex[i]) return;
        }
        D3D11_TEXTURE2D_DESC d[4];
        for (int i = 0; i < 4; ++i) tex[i]->GetDesc(&d[i]);
        const DXGI_FORMAT vf[3] = {ViewFormat(srv[0]), ViewFormat(srv[1]), ViewFormat(srv[3])};
        const UINT w = d[3].Width, h = d[3].Height;
        if (!g_formatsLogged) {
            g_formatsLogged = true;
            Log("DLAA: TAA resolve found: depth %ux%u fmt %d (view %d), color %ux%u fmt %d (view %d), motion %ux%u fmt "
                "%d (view %d), target %ux%u fmt %d",
                d[0].Width, d[0].Height, (int)d[0].Format, (int)vf[0], d[1].Width, d[1].Height, (int)d[1].Format,
                (int)vf[1], d[2].Width, d[2].Height, (int)d[2].Format, (int)vf[2], w, h, (int)d[3].Format);
        }
        for (int i = 0; i < 4; ++i)
            if (d[i].Width != w || d[i].Height != h || d[i].SampleDesc.Count != 1) {
                Log("DLAA: TAA inputs differ in size or are multisampled; the game's TAA for this draw");
                return;
            }
        // DLSS reads typed resources: a typeless engine texture is copied into a typed one first.
        ID3D11Resource* in[3] = {};
        for (int i = 0; i < 3; ++i) {
            if (!IsTypeless(d[i].Format)) {
                in[i] = tex[i];
                continue;
            }
            ID3D11Texture2D* s = ScratchFor(d[i], vf[i], D3D11_BIND_SHADER_RESOURCE);
            if (!s) return;
            c->CopyResource(s, tex[i]);
            in[i] = s;
        }
        ID3D11Texture2D* out = ScratchFor(d[3], d[3].Format, D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE);
        if (!out) return;

        // The projection offset the render tasks set in this frame (SetJitter hook). The engine clears the device
        // slots again before the TAA replay (test 2), so the device array is only a fallback.
        float j[2] = {0, 0};
        void* dev = nullptr;
        uint32_t counter = 0;
        const uint32_t ctx = g_lastCtx.load();
        if (g_devGlobal && hooks::TryRead(g_devGlobal, &dev, sizeof(dev)) && dev) {
            hooks::TryRead((const uint8_t*)dev + kFrameCounterOffset, &counter, sizeof(counter));
            if (g_jitCounter == counter) {
                j[0] = g_jit[0];
                j[1] = g_jit[1];
            } else if (ctx < kMaxCtx) {
                hooks::TryRead((const uint8_t*)dev + kJitterOffset + ctx * 8, j, sizeof(j));
            }
        }
        if (g_diagLogs < 12 && dev) {
            ++g_diagLogs;
            Log("DLAA: evaluation %d: motion vectors %p, frame counter %u, jitter used (%.4f, %.4f); last non-zero "
                "SetJitter (%.4f, %.4f) context %u at frame counter %u; SetJitter calls %llu (non-zero %llu)",
                g_diagLogs, (void*)tex[2], counter, j[0], j[1], g_jit[0], g_jit[1], g_jitCtx, g_jitCounter,
                (unsigned long long)g_setCalls, (unsigned long long)g_setNonzero);
        }

        const uint64_t frame = g_frame.load();
        ID3DDeviceContextState* prev = nullptr;
        g_ctx1->SwapDeviceContextState(g_ctxState, &prev);
        t_inside = true;
        NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
        Feature* f = nullptr;
        Protect("evaluate", [&] {
            f = FeatureFor(c, (uintptr_t)tex[2], w, h);
            if (!f) return;
            NVSDK_NGX_D3D11_DLSS_Eval_Params e{};
            e.Feature.pInColor = in[1];
            e.Feature.pInOutput = out;
            e.pInDepth = in[0];
            e.pInMotionVectors = in[2];
            e.InJitterOffsetX = j[0] * Cfg().dlaaJitterScaleX;
            e.InJitterOffsetY = j[1] * Cfg().dlaaJitterScaleY;
            e.InRenderSubrectDimensions.Width = w;
            e.InRenderSubrectDimensions.Height = h;
            e.InReset = (f->fresh || frame > f->lastFrame + 2) ? 1 : 0;
            // The engine stores uv now - uv before; DLSS wants the offset to the previous position in pixels.
            e.InMVScaleX = -(float)w;
            e.InMVScaleY = -(float)h;
            e.InPreExposure = 1.0f;
            e.InExposureScale = 1.0f;
            r = NGX_D3D11_EVALUATE_DLSS_EXT(c, f->handle, g_params, &e);
        });
        t_inside = false;
        g_ctx1->SwapDeviceContextState(prev, nullptr);
        SafeRelease(prev);
        if (!f || NVSDK_NGX_FAILED(r)) {
            if (f && g_evalFails < 5) Log("DLAA: DLSS evaluate failed: %s", ResultText(r).c_str());
            if (++g_evalFails >= kMaxEvalFails) {
                Log("DLAA: %d failed evaluations in a row; the game's TAA for the rest of the session", g_evalFails);
                SetState("off: DLSS evaluation keeps failing");
                g_failed = true;
                g_on = false;
            }
            return;
        }
        if (g_evalFails || !g_firstEvalLogged) {
            g_firstEvalLogged = true;
            Log("DLAA: DLSS replaces the game's TAA resolve (%ux%u)", w, h);
            SetState("on (DLSS replaces the game's TAA)");
        }
        g_evalFails = 0;
        f->fresh = false;
        f->lastFrame = frame;
        c->CopyResource(tex[3], out);
        result = kDone;
    }();
    for (auto*& t : tex) SafeRelease(t);
    for (auto*& v : srv) SafeRelease(v);
    SafeRelease(rtv);
    return result;
}

#endif  // QVFR_HAVE_DLSS

bool Wanted() { return g_on.load() && g_active.load() && !g_failed; }

void STDMETHODCALLTYPE D_DrawIndexed(ID3D11DeviceContext* c, UINT n, UINT start, INT base) {
    if (n == 6 && !t_inside && Wanted() && c == gpu::Context()) {
        int r = kNotTaa;
#ifdef QVFR_HAVE_DLSS
        Protect("draw", [&] { r = Evaluate(c); });
#endif
        if (r == kDone) {
            g_evals.fetch_add(1, std::memory_order_relaxed);
            g_lastOkFrame = g_frame.load();
            return;
        }
        if (r == kFailed) g_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
    o_drawIndexed(c, n, start, base);
}

void EnsureD3DHooks() {
    if (g_d3dHooked) return;
    ID3D11DeviceContext* imm = gpu::Context();
    if (!imm) return;  // before the first Submit: the game's TAA
    g_d3dHooked = true;
    Protect("D3D11 hooks", [&] {
        void** vt = *(void***)imm;
        bool ok = hooks::CreateAt(vt[12], (void*)D_DrawIndexed, (void**)&o_drawIndexed, "DrawIndexed");
        Log("DLAA: D3D11 DrawIndexed hook %s on immediate context %p", ok ? "installed" : "FAILED", (void*)imm);
        if (!ok) {
            g_failed = true;
            g_on = false;
            SetState("off: D3D11 draw hook failed");
        }
    });
}

// Records the TAA resolve into the engine's command buffer: the render context gives the jitter at replay time.
void __fastcall D_Resolve(void* taa, void* src, void* normal, uint32_t ctx, void* camera, uint64_t a6, int view) {
    g_resolves.fetch_add(1, std::memory_order_relaxed);
    g_lastCtx = ctx;
    if (Wanted()) EnsureD3DHooks();
    o_resolve(taa, src, normal, ctx, camera, a6, view);
}

// The locks pass only feeds the game's TAA: not recorded while DLSS replaced the last frames' resolves.
void __fastcall D_SetJitter(void* dev, const float* offset, uint32_t ctx) {
    o_setJitter(dev, offset, ctx);
    void* ret = _ReturnAddress();
    Protect("SetJitter", [&] {
        float v[2] = {0, 0};
        if (!hooks::TryRead(offset, v, sizeof(v))) return;
        uint32_t counter = 0;
        hooks::TryRead((const uint8_t*)dev + kFrameCounterOffset, &counter, sizeof(counter));
        ++g_setCalls;
        if (v[0] != 0.0f || v[1] != 0.0f) {
            ++g_setNonzero;
            g_jit[0] = v[0];
            g_jit[1] = v[1];
            g_jitCounter = counter;
            g_jitCtx = ctx;
        }
        if (g_setLogs < 40) {
            ++g_setLogs;
            Log("DLAA: SetJitter #%d: context %u, (%.4f, %.4f), frame counter %u, from AMS2AVX.exe+0x%llX", g_setLogs, ctx,
                v[0], v[1], counter, (unsigned long long)((uintptr_t)ret - hooks::ExeBase()));
        }
    });
}

void __fastcall D_Locks(void* taa, void* src, void* normal, uint32_t ctx, void* camera, uint64_t a6, int view) {
    if (Wanted() && Cfg().dlaaSkipLocks && g_evals.load() && g_frame.load() <= g_lastOkFrame.load() + 1) {
        g_locksSkipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    o_locks(taa, src, normal, ctx, camera, a6, view);
}

}  // namespace

void Install() {
    if (!Cfg().dlaa) return;
    if (!hooks::Init()) return;
    static bool first = true;
    if (!first) {
        if (g_installed && !g_failed) {
            hooks::EnableAll();
            g_on = true;
        }
        return;
    }
    first = false;
#ifndef QVFR_HAVE_DLSS
    Log("DLAA: this proxy was built without the NVIDIA DLSS SDK; the game's TAA stays");
    SetState("off: this build has no DLSS");
    return;
#else
    if (Cfg().quadMode == QuadMode::Probe) {
        Log("DLAA: off with [quad] mode=probe (both hook the D3D11 draws)");
        SetState("off with [quad] mode=probe");
        return;
    }
    if (Cfg().quadMode == QuadMode::Quad) {
        // The focus groups run as view slot 2, so they would share the right eye's DLSS history.
        Log("DLAA: off with [quad] mode=quad (the prototype works in stereo only); the game's TAA stays");
        SetState("off with [quad] mode=quad (stereo only for now)");
        return;
    }
    if (!hooks::EngineAvailable()) {
        Log("DLAA: exe not supported (see the Hooks lines); the game's TAA stays");
        SetState("off: unsupported game build");
        return;
    }
    if (sysinfo::GameSetting("temporalantialiasenabled") != "1")
        diag::Report(diag::Level::Warning, "dlaa-taa",
                     "[dlaa] mode=dlaa replaces the game's TAA, but TAA is off in the game's graphics options. Turn "
                     "TAA on (and MSAA off) to use DLAA.");
    std::error_code ec;
    if (!fs::exists(ModuleDir() / L"qvfr" / L"nvngx_dlss.dll", ec)) {
        diag::Report(diag::Level::Warning, "dlaa-dll",
                     "[dlaa] mode=dlaa needs x64\\qvfr\\nvngx_dlss.dll (NVIDIA DLSS); it is missing, so the game's TAA "
                     "stays.");
        SetState("off: x64\\qvfr\\nvngx_dlss.dll missing");
        return;
    }
    Protect("Install", [&] {
        const uint8_t* load = (const uint8_t*)hooks::Address(Anchor::TaaResolveDeviceLoad);
        if (!load || load[0] != 0x48 || load[1] != 0x8B || load[2] != 0x0D) {
            Log("DLAA: device load does not match; no DLAA");
            return;
        }
        g_devGlobal = (void**)(load + 7 + *(const int32_t*)(load + 3));
        bool ok = hooks::Create(Anchor::TaaResolve, (void*)D_Resolve, (void**)&o_resolve);
        ok &= hooks::Create(Anchor::TaaLocks, (void*)D_Locks, (void**)&o_locks);
        ok &= hooks::Create(Anchor::SetJitter, (void*)D_SetJitter, (void**)&o_setJitter);
        g_installed = ok;
        g_on = ok;
        Log("DLAA: engine hooks %s (CTAAControl resolve and locks; device pointer at AMS2AVX.exe+0x%llX)",
            ok ? "installed" : "INCOMPLETE; the game's TAA stays",
            (unsigned long long)((uintptr_t)g_devGlobal - hooks::ExeBase()));
        SetState(ok ? "on (waiting for the first TAA frame)" : "off: engine hooks failed");
    });
    g_active = true;
#endif
}

void OnFrameStart() {
    if (!g_installed) return;
    g_frame.fetch_add(1);
    Protect("frame", [&] {
        const Config& c = Cfg();
        if (Pressed(c.dlaaKeyMod, c.dlaaKey, g_keyDown)) {
            const bool on = !g_active.load();
            g_active = on;
            Log("DLAA: hotkey: %s", on ? "DLAA" : "the game's TAA");
#ifdef QVFR_HAVE_DLSS
            if (on) ResetAllLocked();
#endif
        }
    });
}

void LogStats(bool totals) {
    if (!g_installed) return;
    const uint64_t e = g_evals.load(), f = g_fallbacks.load(), l = g_locksSkipped.load(), r = g_resolves.load();
    const uint64_t sc = g_setCalls, sn = g_setNonzero;
    Log("DLAA%s: %llu DLSS evaluations, %llu TAA draws left to the game, %llu locks passes skipped, %llu resolves "
        "recorded, %llu SetJitter calls (%llu non-zero, last (%.4f, %.4f)); %s, %s",
        totals ? " (session total)" : "", (unsigned long long)(totals ? e : e - g_lastEvals),
        (unsigned long long)(totals ? f : f - g_lastFallbacks), (unsigned long long)(totals ? l : l - g_lastLocks),
        (unsigned long long)(totals ? r : r - g_lastResolves), (unsigned long long)(totals ? sc : sc - g_lastSetCalls),
        (unsigned long long)(totals ? sn : sn - g_lastSetNonzero), g_jit[0], g_jit[1],
        g_active.load() ? "hotkey: DLAA" : "hotkey: game TAA", Describe().c_str());
    g_lastEvals = e;
    g_lastFallbacks = f;
    g_lastLocks = l;
    g_lastResolves = r;
    g_lastSetCalls = sc;
    g_lastSetNonzero = sn;
}

std::string Describe() {
    std::lock_guard lock(g_sm);
    return "DLAA: " + g_state;
}

void Shutdown() {
    if (!g_installed) return;
    g_on = false;
#ifdef QVFR_HAVE_DLSS
    Protect("Shutdown", [&] { DropNgxLocked(); });
    g_ngxLogLines = 0;
#endif
    hooks::DisableAll();
}

}  // namespace qvfr::dlaa
