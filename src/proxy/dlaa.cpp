#include "dlaa.h"

#include <d3d11_1.h>

#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "common.h"
#include "diag.h"
#include "gpu.h"
#include "hooks.h"
#include "mvr_motion.h"
#include "sun_fx.h"
#include "sysinfo.h"

#ifdef QVFR_HAVE_DLSS
#include <nvsdk_ngx_helpers_d3d.h>

#include "dlaa_cockpit_cs.h"  // kDlaaCockpitCS (fxc from shaders/dlaa.hlsl)
#include "dlaa_mask_cs.h"     // kDlaaMaskCS
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
bool g_external = false;             // quad mode: the draws come from sun_fx's DrawIndexed hook (Replace)
// Quad eye group being replayed (-1 = none, stereo): its own DLSS feature. Index group + 1 in the per-view arrays.
std::atomic<int> g_group{-1};
constexpr int kViews = 5;
constexpr const char* kViewName[kViews] = {"eye", "outer L", "outer R", "focus L", "focus R"};
std::atomic<uint64_t> g_viewEvals[kViews];
uint64_t g_lastViewEvals[kViews] = {};
bool g_viewLogged[kViews] = {};  // first DLSS evaluation of the view logged
bool g_keyDown = false;
bool g_expKeyDown = false;
bool g_presetKeyDown = false;
int CockpitTaaNow() { return Cfg().dlaaCockpitTaa; }  // displays/cockpit from the game's TAA (ini, read live)
const char* CockpitText(int mode) {
    return mode == 0 ? "nothing" : mode == 1 ? "displays and mirrors" : "displays, mirrors and the player's car";
}
thread_local bool t_after = false;  // the forwarded draw is the game's TAA resolve: DLSS and the cockpit after it
// Exposure hotkey ([dlaa] exposure_key): step 0 = the ini settings, then DLSS auto exposure scaled down and fixed
// exposures (dlaa-plan D4, test 4: the lower the exposure, the less DLSS dims the lights).
struct ExpStep {
    bool autoExp;
    float value;  // fixed exposure (auto off) or scale (auto on)
};
constexpr ExpStep kExpSteps[] = {{false, 0}, {true, 1.0f}, {false, 0.005f}, {false, 0.002f}, {false, 0.001f}};
constexpr int kExpStepCount = (int)(sizeof(kExpSteps) / sizeof(kExpSteps[0]));
std::atomic<int> g_expStep{0};
bool AutoExposureNow() {
    const int s = g_expStep.load();
    return s == 0 ? Cfg().dlaaAutoExposure : kExpSteps[s].autoExp;
}
float ExposureNow() {  // fixed exposure, 0 = none (DLSS auto exposure or its default 1)
    const int s = g_expStep.load();
    if (s == 0) return Cfg().dlaaAutoExposure ? 0.0f : Cfg().dlaaExposure;
    return kExpSteps[s].autoExp ? 0.0f : kExpSteps[s].value;
}
float ExposureScaleNow() {
    const int s = g_expStep.load();
    if (s == 0) return Cfg().dlaaExposureScale;
    return kExpSteps[s].autoExp ? kExpSteps[s].value : 1.0f;
}
// Preset hotkey ([dlaa] preset_key): step 0 = the ini preset, then J, K, M, L.
constexpr int kPresetSteps[] = {-1, 10, 11, 13, 12};
constexpr int kPresetStepCount = (int)(sizeof(kPresetSteps) / sizeof(kPresetSteps[0]));
std::atomic<int> g_presetStep{0};
int PresetNow() {
    const int s = g_presetStep.load();
    return s == 0 ? Cfg().dlaaPreset : kPresetSteps[s];
}
std::string ExposureText() {
    char buf[96];
    const float e = ExposureNow(), k = ExposureScaleNow();
    const int p = PresetNow();
    char preset[16];
    snprintf(preset, sizeof(preset), p > 0 ? "%c" : "default", 'a' + p - 1);
    if (AutoExposureNow())
        snprintf(buf, sizeof(buf), "exposure auto x %g, preset %s", k, preset);
    else if (e > 0)
        snprintf(buf, sizeof(buf), "exposure %g x %g, preset %s", e, k, preset);
    else
        snprintf(buf, sizeof(buf), "exposure 1 (DLSS default) x %g, preset %s", k, preset);
    return buf;
}
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
std::atomic<int> g_traceLeft{0};  // TraceJitter: further SetJitter calls to log
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
    int view = 0;       // index in kViewName (quad eye group + 1)
    uintptr_t key = 0;  // the motion vector texture
    UINT w = 0, h = 0;
    int flags = 0;
    int preset = -1;
    uint64_t lastFrame = 0;
    bool fresh = true;
};

struct Scratch {
    ID3D11Texture2D* tex = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;  // made on first use (UavFor)
    ID3D11ShaderResourceView* srv = nullptr;   // made on first use (SrvFor)
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
bool g_formatsLogged[kViews] = {};
// Bias-current-color mask (shaders/dlaa.hlsl) and the fixed exposure texture ([dlaa] bias_mask, exposure).
ID3D11ComputeShader* g_maskCs = nullptr;
ID3D11ComputeShader* g_cockpitCs = nullptr;
ID3D11Buffer* g_cockpitCb = nullptr;
uint32_t g_cockpitCbBits = 0;
uint64_t g_cockpitEvals = 0;
ID3D11Buffer* g_maskCb = nullptr;
float g_maskCbValue = -1;
bool g_maskFailed = false;
ID3D11Texture2D* g_expTex = nullptr;
float g_expValue = -1;
uint64_t g_maskEvals = 0, g_expEvals = 0;

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
    for (Scratch& s : g_scratch) {
        SafeRelease(s.srv);
        SafeRelease(s.uav);
        SafeRelease(s.tex);
    }
    g_scratch.clear();
    mvrmotion::Release();
    SafeRelease(g_cockpitCs);
    SafeRelease(g_cockpitCb);
    g_cockpitCbBits = 0;
    SafeRelease(g_maskCs);
    SafeRelease(g_maskCb);
    g_maskCbValue = -1;
    g_maskFailed = false;
    SafeRelease(g_expTex);
    g_expValue = -1;
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
    if (g_scratch.size() >= 12) {
        SafeRelease(g_scratch.front().srv);
        SafeRelease(g_scratch.front().uav);
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

ID3D11UnorderedAccessView* UavFor(ID3D11Texture2D* tex) {
    for (Scratch& s : g_scratch) {
        if (s.tex != tex) continue;
        if (!s.uav && FAILED(g_dev->CreateUnorderedAccessView(tex, nullptr, &s.uav))) {
            Log("DLAA: UAV for texture %ux%u format %d failed", s.w, s.h, (int)s.fmt);
            s.uav = nullptr;
        }
        return s.uav;
    }
    return nullptr;
}

ID3D11ShaderResourceView* SrvFor(ID3D11Texture2D* tex) {
    for (Scratch& s : g_scratch) {
        if (s.tex != tex) continue;
        if (!s.srv && FAILED(g_dev->CreateShaderResourceView(tex, nullptr, &s.srv))) {
            Log("DLAA: SRV for texture %ux%u format %d failed", s.w, s.h, (int)s.fmt);
            s.srv = nullptr;
        }
        return s.srv;
    }
    return nullptr;
}

bool EnsureCockpitShader() {
    if (g_cockpitCs && g_cockpitCb) return true;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 16;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(g_dev->CreateComputeShader(kDlaaCockpitCS, sizeof(kDlaaCockpitCS), nullptr, &g_cockpitCs)) ||
        FAILED(g_dev->CreateBuffer(&bd, nullptr, &g_cockpitCb))) {
        SafeRelease(g_cockpitCs);
        SafeRelease(g_cockpitCb);
        return false;
    }
    g_cockpitCbBits = 0;
    return true;
}

bool EnsureMaskShader() {
    if (g_maskCs && g_maskCb) return true;
    if (g_maskFailed) return false;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 16;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(g_dev->CreateComputeShader(kDlaaMaskCS, sizeof(kDlaaMaskCS), nullptr, &g_maskCs)) ||
        FAILED(g_dev->CreateBuffer(&bd, nullptr, &g_maskCb))) {
        Log("DLAA: the bias mask shader could not be made; no mask");
        g_maskFailed = true;
        SafeRelease(g_maskCs);
        SafeRelease(g_maskCb);
        return false;
    }
    g_maskCbValue = -1;
    return true;
}

// A 1 x 1 R32_FLOAT texture with the fixed DLSS exposure ([dlaa] exposure with auto_exposure=0).
ID3D11Texture2D* ExposureTexture(ID3D11DeviceContext* c, float value) {
    if (!g_expTex) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = d.Height = d.MipLevels = d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R32_FLOAT;
        d.SampleDesc.Count = 1;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &g_expTex))) {
            g_expTex = nullptr;
            return nullptr;
        }
        g_expValue = -1;
    }
    if (value != g_expValue) {
        c->UpdateSubresource(g_expTex, 0, nullptr, &value, sizeof(float), 0);
        g_expValue = value;
        Log("DLAA: fixed DLSS exposure %.5f", value);
    }
    return g_expTex;
}

ID3D11Texture2D* TexOf(ID3D11View* v) {
    if (!v) return nullptr;
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    if (!r) return nullptr;
    ID3D11Texture2D* t = nullptr;
    if (FAILED(r->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t))) t = nullptr;
    r->Release();
    return t;
}

// ---------------------------------------------------------------- shot-frame measurement
// In a quad shot frame (right Ctrl + F10) each view's TAA resolve is measured once: luminance of the input color and of
// the output (DLSS or the game's TAA), the brightest input pixels and the stencil bits the game's TAA reads. Shows
// whether DLSS loses highlight energy (dlaa-plan D4, test 2). Stalls the frame (read back), so only then.
std::atomic<bool> g_shot{false};
unsigned g_shotViews = 0;  // views measured in this shot frame (bit per view)
struct PendingTaa {        // the game's TAA resolve of a shot frame: its target is read at the next draw
    ID3D11Texture2D* in = nullptr;
    ID3D11Texture2D* stencil = nullptr;
    ID3D11Texture2D* target = nullptr;
    int view = 0;
};
PendingTaa g_pend;

bool ShotWanted(int view) { return g_shot.load(std::memory_order_relaxed) && !(g_shotViews & (1u << view)); }

ID3D11Texture2D* StagingCopy(ID3D11DeviceContext* c, ID3D11Texture2D* src) {
    if (!src) return nullptr;
    D3D11_TEXTURE2D_DESC d;
    src->GetDesc(&d);
    if (d.SampleDesc.Count != 1) return nullptr;
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    ID3D11Texture2D* s = nullptr;
    if (!dev || FAILED(dev->CreateTexture2D(&d, nullptr, &s))) s = nullptr;
    SafeRelease(dev);
    if (s) c->CopyResource(s, src);
    return s;
}

float Float11(uint32_t v) {
    const uint32_t e = v >> 6, m = v & 63;
    if (e == 0) return std::ldexp((float)m / 64.0f, -14);
    if (e == 31) return 65024.0f;
    return std::ldexp(1.0f + (float)m / 64.0f, (int)e - 15);
}

float Float10(uint32_t v) {
    const uint32_t e = v >> 5, m = v & 31;
    if (e == 0) return std::ldexp((float)m / 32.0f, -14);
    if (e == 31) return 64512.0f;
    return std::ldexp(1.0f + (float)m / 32.0f, (int)e - 15);
}

float LumaR11G11B10(uint32_t p) {
    return 0.2126f * Float11(p & 0x7FF) + 0.7152f * Float11((p >> 11) & 0x7FF) + 0.0722f * Float10(p >> 22);
}

struct LumStats {
    static constexpr int kBins = 400;  // log2 luminance from -25 to +25, 8 bins per stop
    double sum = 0;
    uint64_t n = 0;
    float max = 0;
    std::vector<uint32_t> hist = std::vector<uint32_t>(kBins);
    void Add(float l) {
        sum += l;
        ++n;
        max = std::max(max, l);
        const int b = l > 0 ? (int)std::floor((std::log2(l) + 25.0f) * 8.0f) : 0;
        ++hist[std::clamp(b, 0, kBins - 1)];
    }
    float Mean() const { return n ? (float)(sum / (double)n) : 0; }
    float Pct(double p) const {  // upper edge of the bin holding the p-quantile
        uint64_t want = (uint64_t)(p * (double)n), seen = 0;
        for (int b = 0; b < kBins; ++b) {
            seen += hist[b];
            if (seen > want) return std::exp2((float)(b + 1) / 8.0f - 25.0f);
        }
        return max;
    }
};

void MeasureShot(ID3D11DeviceContext* c, int view, const char* mode, ID3D11Texture2D* sIn, ID3D11Texture2D* sSt,
                 ID3D11Texture2D* sOut) {
    if (!sIn || !sOut) return;
    D3D11_TEXTURE2D_DESC di, dout, ds{};
    sIn->GetDesc(&di);
    sOut->GetDesc(&dout);
    if (di.Format != DXGI_FORMAT_R11G11B10_FLOAT || dout.Format != DXGI_FORMAT_R11G11B10_FLOAT ||
        di.Width != dout.Width || di.Height != dout.Height) {
        Log("DLAA shot %s (%s): formats or sizes not as expected (%d %ux%u, %d %ux%u)", kViewName[view], mode,
            (int)di.Format, di.Width, di.Height, (int)dout.Format, dout.Width, dout.Height);
        return;
    }
    bool stencilOk = false;
    if (sSt) {
        sSt->GetDesc(&ds);
        stencilOk = ds.Format >= DXGI_FORMAT_R16G16B16A16_TYPELESS && ds.Format <= DXGI_FORMAT_R16G16B16A16_SINT &&
                    ds.Width == di.Width && ds.Height == di.Height;
    }
    D3D11_MAPPED_SUBRESOURCE mi{}, mo{}, ms{};
    if (FAILED(c->Map(sIn, 0, D3D11_MAP_READ, 0, &mi))) return;
    if (FAILED(c->Map(sOut, 0, D3D11_MAP_READ, 0, &mo))) {
        c->Unmap(sIn, 0);
        return;
    }
    if (stencilOk && FAILED(c->Map(sSt, 0, D3D11_MAP_READ, 0, &ms))) stencilOk = false;
    const UINT w = di.Width, h = di.Height;
    auto pin = [&](UINT x, UINT y) { return ((const uint32_t*)((const uint8_t*)mi.pData + (size_t)y * mi.RowPitch))[x]; };
    auto pout = [&](UINT x, UINT y) { return ((const uint32_t*)((const uint8_t*)mo.pData + (size_t)y * mo.RowPitch))[x]; };
    auto sten = [&](UINT x, UINT y) {
        return (uint32_t)((const uint16_t*)((const uint8_t*)ms.pData + (size_t)y * ms.RowPitch))[x * 4 + 3] >> 8;
    };
    LumStats li, lo;
    for (UINT y = 0; y < h; ++y)
        for (UINT x = 0; x < w; ++x) {
            li.Add(LumaR11G11B10(pin(x, y)));
            lo.Add(LumaR11G11B10(pout(x, y)));
        }
    const float t99 = li.Pct(0.99), t999 = li.Pct(0.999);
    double b99i = 0, b99o = 0, b999i = 0, b999o = 0, s2i = 0, s2o = 0, d2 = 0, dn = 0, sni = 0;
    std::vector<uint32_t> valCount(stencilOk ? 65536 : 0);
    std::vector<double> valLum(stencilOk ? 65536 : 0);
    uint64_t n2 = 0, n4 = 0;
    for (UINT y = 0; y < h; ++y)
        for (UINT x = 0; x < w; ++x) {
            const float a = LumaR11G11B10(pin(x, y)), b = LumaR11G11B10(pout(x, y));
            if (a >= t99) {
                b99i += a;
                b99o += b;
            }
            if (a >= t999) {
                b999i += a;
                b999o += b;
            }
            if (!stencilOk) continue;
            const uint32_t full = ((const uint16_t*)((const uint8_t*)ms.pData + (size_t)y * ms.RowPitch))[x * 4 + 3];
            ++valCount[full];
            valLum[full] += a;
            const uint32_t s = sten(x, y);
            if (s & 2) {
                ++n2;
                s2i += a;
                s2o += b;
                d2 += std::fabs(b - a);
            } else {
                dn += std::fabs(b - a);
                sni += a;
            }
            if (s & 4) ++n4;
        }
    if (stencilOk) c->Unmap(sSt, 0);
    c->Unmap(sOut, 0);
    c->Unmap(sIn, 0);
    const double px = (double)w * h;
    auto ratio = [](double o, double i) { return i > 0 ? o / i : 0.0; };
    Log("DLAA shot %s (%s, %ux%u): input luminance mean %.4f, p50 %.4f, p99 %.3f, p99.9 %.3f, max %.1f; output mean %.4f "
        "(output/input %.3f), p50 %.4f, p99 %.3f, p99.9 %.3f, max %.1f; brightest 1%% of the input pixels: output/input "
        "%.3f, brightest 0.1%%: %.3f",
        kViewName[view], mode, w, h, li.Mean(), li.Pct(0.5), t99, t999, li.max, lo.Mean(), ratio(lo.Mean(), li.Mean()),
        lo.Pct(0.5), lo.Pct(0.99), lo.Pct(0.999), lo.max, ratio(b99o, b99i), ratio(b999o, b999i));
    if (stencilOk)
        Log("DLAA shot %s (%s): stencil bit 0x200 (the game's short TAA history) %.1f%% of the pixels, their output/input "
            "%.3f, their mean |output - input| / mean input %.3f (other pixels %.3f); bit 0x400 %.1f%%",
            kViewName[view], mode, 100.0 * (double)n2 / px, ratio(s2o, s2i), ratio(d2, s2i), ratio(dn, sni),
            100.0 * (double)n4 / px);
    else
        Log("DLAA shot %s (%s): no stencil statistics (format %d)", kViewName[view], mode, (int)ds.Format);
    if (stencilOk) {
        // The most common full stencil values (high byte = class: 0x86 the player's car, 0x88 displays and mirrors).
        std::vector<uint32_t> order;
        for (uint32_t v = 0; v < 65536; ++v)
            if (valCount[v]) order.push_back(v);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return valCount[a] > valCount[b]; });
        std::string list;
        for (size_t i = 0; i < order.size() && i < 16; ++i) {
            const uint32_t v = order[i];
            char buf[96];
            snprintf(buf, sizeof(buf), "%s0x%04X %.2f%% (input %.3f)", i ? "; " : "", v,
                     100.0 * (double)valCount[v] / px, valLum[v] / (double)valCount[v]);
            list += buf;
        }
        Log("DLAA shot %s: %zu stencil values, the most common: %s", kViewName[view], order.size(), list.c_str());
    }
}

void ReleasePending() {
    SafeRelease(g_pend.in);
    SafeRelease(g_pend.stencil);
    SafeRelease(g_pend.target);
}

// One DLSS feature per view, motion vector texture (the engine keeps one per view slot; taa_views adds the focus ones)
// and size. The view keeps the focus histories apart even when the focus groups share a texture (taa_views off).
Feature* FeatureFor(ID3D11DeviceContext* c, int view, uintptr_t key, UINT w, UINT h) {
    const Config& cfg = Cfg();
    int flags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (cfg.dlaaMvJittered) flags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
    if (AutoExposureNow()) flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    const int preset = PresetNow();
    Feature* f = nullptr;
    for (Feature& x : g_feat)
        if (x.view == view && x.key == key && x.w == w && x.h == h) f = &x;
    if (f && f->handle && f->flags == flags && f->preset == preset) return f;
    if (!f) {
        if (g_feat.size() >= 8) {
            if (g_feat.front().handle) NVSDK_NGX_D3D11_ReleaseFeature(g_feat.front().handle);
            g_feat.erase(g_feat.begin());
        }
        g_feat.push_back(Feature{});
        f = &g_feat.back();
        f->view = view;
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
        Log("DLAA: DLSS feature for %s, motion vectors %p (%ux%u, flags 0x%X, preset %d) failed: %s", kViewName[view],
            (void*)key, w, h, flags, preset, ResultText(r).c_str());
        f->handle = nullptr;
        return nullptr;
    }
    f->flags = flags;
    f->preset = preset;
    f->fresh = true;
    Log("DLAA: DLSS feature for %s, motion vectors %p created (%ux%u, DLAA, flags 0x%X, preset %d)", kViewName[view],
        (void*)key, w, h, flags, preset);
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

bool IsTaaResolve(ID3D11RenderTargetView* rtv, ID3D11ShaderResourceView* const* srv) {
    if (!rtv || !srv[0] || !srv[1] || !srv[3]) return false;
    D3D11_RENDER_TARGET_VIEW_DESC rd;
    rtv->GetDesc(&rd);
    return rd.Format == DXGI_FORMAT_R11G11B10_FLOAT && ViewFormat(srv[2]) == DXGI_FORMAT_R11G11B10_FLOAT &&
           ViewFormat(srv[3]) == DXGI_FORMAT_R16G16_FLOAT && ViewFormat(srv[5]) == DXGI_FORMAT_R8_UNORM;
}

// The game's TAA resolve in a shot frame while the hotkey gives the game's TAA: input and stencil now, the target after
// the draw (FinishTaaShot at the next draw).
void StartTaaShot(ID3D11DeviceContext* c) {
    const int view = g_group.load(std::memory_order_relaxed) + 1;
    if (!ShotWanted(view) || g_pend.target) return;
    ID3D11ShaderResourceView* srv[6] = {};
    c->PSGetShaderResources(0, 6, srv);
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    if (IsTaaResolve(rtv, srv)) {
        ID3D11Texture2D* color = TexOf(srv[1]);
        ID3D11Texture2D* st = TexOf(srv[4]);
        g_pend.in = StagingCopy(c, color);
        g_pend.stencil = StagingCopy(c, st);
        g_pend.target = TexOf(rtv);
        g_pend.view = view;
        g_shotViews |= 1u << view;
        SafeRelease(color);
        SafeRelease(st);
        if (!g_pend.in || !g_pend.target) ReleasePending();
    }
    for (auto*& v : srv) SafeRelease(v);
    SafeRelease(rtv);
}

void FinishTaaShot(ID3D11DeviceContext* c) {
    ID3D11Texture2D* out = StagingCopy(c, g_pend.target);
    MeasureShot(c, g_pend.view, "game TAA", g_pend.in, g_pend.stencil, out);
    SafeRelease(out);
    ReleasePending();
}

// A DrawIndexed(6) on the immediate context. The engine records its TAA resolve into a command buffer and replays it
// later, so the draw is recognized by what is bound (docs/ams2-renderer.md): target and history R11G11B10_FLOAT,
// t3 motion vectors R16G16_FLOAT, t5 locks R8_UNORM. Returns kNotTaa, kDone (DLSS ran, skip the draw) or kFailed.
// afterDraw: the game's TAA resolve already ran (cockpit_taa): DLSS runs after it and only the pixels outside the
// player's car (stencil bit 0x200) take the DLSS result.
int Evaluate(ID3D11DeviceContext* c, bool afterDraw) {
    const int view = g_group.load(std::memory_order_relaxed) + 1;
    ID3D11ShaderResourceView* srv[6] = {};
    c->PSGetShaderResources(0, 6, srv);
    ID3D11RenderTargetView* rtv = nullptr;
    c->OMGetRenderTargets(1, &rtv, nullptr);
    ID3D11Texture2D* tex[4] = {};  // depth, color, motion vectors, target
    int result = kNotTaa;
    ID3D11Texture2D* stencil = nullptr;  // t4: the game's normal/stencil (bias mask, shot statistics)
    [&] {
        if (!IsTaaResolve(rtv, srv)) return;
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
        if (!g_formatsLogged[view]) {
            g_formatsLogged[view] = true;
            Log("DLAA: TAA resolve found (%s): depth %ux%u fmt %d (view %d), color %ux%u fmt %d (view %d), motion %ux%u fmt "
                "%d (view %d), target %ux%u fmt %d",
                kViewName[view], d[0].Width, d[0].Height, (int)d[0].Format, (int)vf[0], d[1].Width, d[1].Height, (int)d[1].Format,
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
        stencil = TexOf(srv[4]);

        // Bias-current-color mask where the game's TAA keeps a short history (stencil bit 0x200, shaders/dlaa.hlsl).
        const Config& cfg = Cfg();
        ID3D11Texture2D* mask = nullptr;
        ID3D11UnorderedAccessView* maskUav = nullptr;
        if (cfg.dlaaBiasMask > 0 && stencil && EnsureMaskShader()) {
            D3D11_TEXTURE2D_DESC sd;
            stencil->GetDesc(&sd);
            if (sd.Width == w && sd.Height == h && sd.SampleDesc.Count == 1) {
                mask = ScratchFor(d[3], DXGI_FORMAT_R8_UNORM, D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE);
                maskUav = mask ? UavFor(mask) : nullptr;
            }
            if (!maskUav) {
                mask = nullptr;
            } else if (cfg.dlaaBiasMask != g_maskCbValue) {
                const float v[4] = {cfg.dlaaBiasMask, 0, 0, 0};
                c->UpdateSubresource(g_maskCb, 0, nullptr, v, 0, 0);
                g_maskCbValue = cfg.dlaaBiasMask;
            }
        }
        ID3D11Texture2D* expTex =
            ExposureNow() > 0 ? ExposureTexture(c, ExposureNow()) : nullptr;
        // Cockpit from the game's TAA: its result (now in the target) is copied aside before DLSS overwrites anything.
        ID3D11ShaderResourceView* taaSrv = nullptr;
        ID3D11UnorderedAccessView* outUav = nullptr;
        if (afterDraw) {
            D3D11_TEXTURE2D_DESC sd{};
            if (stencil) stencil->GetDesc(&sd);
            ID3D11Texture2D* taaCopy = ScratchFor(d[3], d[3].Format, D3D11_BIND_SHADER_RESOURCE);
            if (!stencil || sd.Width != w || sd.Height != h || !taaCopy || !EnsureCockpitShader()) return;
            taaSrv = SrvFor(taaCopy);
            outUav = UavFor(out);
            if (!taaSrv || !outUav) return;  // the game's TAA result stays everywhere
            c->CopyResource(taaCopy, tex[3]);
            const uint32_t bits = CockpitTaaNow() >= 2 ? 0x0A : 0x08;  // stencil high byte: 0x08 displays, 0x02 car
            if (bits != g_cockpitCbBits) {
                const uint32_t v[4] = {bits, 0, 0, 0};
                c->UpdateSubresource(g_cockpitCb, 0, nullptr, v, 0, 0);
                g_cockpitCbBits = bits;
            }
        }

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
            f = FeatureFor(c, view, (uintptr_t)tex[2], w, h);
            if (!f) return;
            // MVR (mvr_motion.h): outer's motion vectors kept per eye, the focus views' made from them.
            ID3D11Resource* mv = in[2];
            if (view >= 1) {
                const int eye = (view - 1) & 1;
                const bool shot = g_shot.load(std::memory_order_relaxed);
                if (view <= 2)
                    mvrmotion::StoreOuter(c, eye, (ID3D11Texture2D*)in[2], (ID3D11Texture2D*)in[0], frame, shot);
                else if (ID3D11Texture2D* m = mvrmotion::ForFocus(c, eye, (ID3D11Texture2D*)in[0],
                                                                  (ID3D11Texture2D*)in[2], w, h, frame, shot))
                    mv = m;
            }
            if (mask) {
                c->CSSetShader(g_maskCs, nullptr, 0);
                c->CSSetConstantBuffers(0, 1, &g_maskCb);
                c->CSSetShaderResources(0, 1, &srv[4]);
                c->CSSetUnorderedAccessViews(0, 1, &maskUav, nullptr);
                c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
                ID3D11ShaderResourceView* noSrv = nullptr;
                ID3D11UnorderedAccessView* noUav = nullptr;
                c->CSSetShaderResources(0, 1, &noSrv);
                c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
                c->CSSetShader(nullptr, nullptr, 0);
                ++g_maskEvals;
            }
            if (expTex) ++g_expEvals;
            NVSDK_NGX_D3D11_DLSS_Eval_Params e{};
            e.Feature.pInColor = in[1];
            e.Feature.pInOutput = out;
            e.pInDepth = in[0];
            e.pInMotionVectors = mv;
            e.InJitterOffsetX = j[0] * Cfg().dlaaJitterScaleX;
            e.InJitterOffsetY = j[1] * Cfg().dlaaJitterScaleY;
            e.InRenderSubrectDimensions.Width = w;
            e.InRenderSubrectDimensions.Height = h;
            e.InReset = (f->fresh || frame > f->lastFrame + 2) ? 1 : 0;
            // The engine stores uv now - uv before; DLSS wants the offset to the previous position in pixels.
            e.InMVScaleX = -(float)w;
            e.InMVScaleY = -(float)h;
            e.InPreExposure = 1.0f;
            e.InExposureScale = ExposureScaleNow();
            e.pInBiasCurrentColorMask = mask;
            e.pInExposureTexture = expTex;
            r = NGX_D3D11_EVALUATE_DLSS_EXT(c, f->handle, g_params, &e);
            if (afterDraw && NVSDK_NGX_SUCCEED(r)) {
                ID3D11ShaderResourceView* in2[2] = {srv[4], taaSrv};
                c->CSSetShader(g_cockpitCs, nullptr, 0);
                c->CSSetConstantBuffers(1, 1, &g_cockpitCb);
                c->CSSetShaderResources(0, 2, in2);
                c->CSSetUnorderedAccessViews(0, 1, &outUav, nullptr);
                c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
                ID3D11ShaderResourceView* noSrv[2] = {};
                ID3D11UnorderedAccessView* noUav = nullptr;
                c->CSSetShaderResources(0, 2, noSrv);
                c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
                c->CSSetShader(nullptr, nullptr, 0);
                ++g_cockpitEvals;
            }
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
        if (g_evalFails || !g_viewLogged[view]) {
            g_viewLogged[view] = true;
            Log("DLAA: DLSS replaces the game's TAA resolve (%s, %ux%u)", kViewName[view], w, h);
            SetState(view ? "on in quad views (DLSS replaces the game's TAA)" : "on (DLSS replaces the game's TAA)");
        }
        g_viewEvals[view].fetch_add(1, std::memory_order_relaxed);
        g_evalFails = 0;
        f->fresh = false;
        f->lastFrame = frame;
        if (ShotWanted(view)) {
            g_shotViews |= 1u << view;
            ID3D11Texture2D* sIn = StagingCopy(c, (ID3D11Texture2D*)in[1]);
            ID3D11Texture2D* sSt = StagingCopy(c, stencil);
            ID3D11Texture2D* sOut = StagingCopy(c, out);
            char mode[192];
            snprintf(mode, sizeof(mode), "DLAA, game's TAA for %s, %s, bias mask %.2f",
                     CockpitText(afterDraw ? CockpitTaaNow() : 0), ExposureText().c_str(), mask ? cfg.dlaaBiasMask : 0.0f);
            MeasureShot(c, view, mode, sIn, sSt, sOut);
            SafeRelease(sIn);
            SafeRelease(sSt);
            SafeRelease(sOut);
        }
        c->CopyResource(tex[3], out);
        result = kDone;
    }();
    for (auto*& t : tex) SafeRelease(t);
    SafeRelease(stencil);
    for (auto*& v : srv) SafeRelease(v);
    SafeRelease(rtv);
    return result;
}

#endif  // QVFR_HAVE_DLSS

bool Wanted() { return g_on.load() && g_active.load() && !g_failed; }

// A DrawIndexed on some context: true = DLSS replaced the game's TAA resolve (the draw is not forwarded).
bool Replace(ID3D11DeviceContext* c, UINT n) {
    if (t_inside || c != gpu::Context()) return false;
#ifdef QVFR_HAVE_DLSS
    if (g_pend.target) Protect("shot", [&] { FinishTaaShot(c); });  // the last draw was the game's TAA resolve
    if (n == 6 && !g_failed && g_on.load() && !g_active.load() && g_shot.load(std::memory_order_relaxed))
        Protect("shot", [&] { StartTaaShot(c); });
#endif
    if (n != 6 || !Wanted()) return false;
    int r = kNotTaa;
#ifdef QVFR_HAVE_DLSS
    if (CockpitTaaNow()) {
        // The game's TAA draw goes on; DLSS and the cockpit composition run in After.
        Protect("draw", [&] {
            ID3D11ShaderResourceView* srv[6] = {};
            c->PSGetShaderResources(0, 6, srv);
            ID3D11RenderTargetView* rtv = nullptr;
            c->OMGetRenderTargets(1, &rtv, nullptr);
            t_after = IsTaaResolve(rtv, srv);
            for (auto*& v : srv) SafeRelease(v);
            SafeRelease(rtv);
        });
        return false;
    }
    Protect("draw", [&] { r = Evaluate(c, false); });
#endif
    if (r == kDone) {
        g_evals.fetch_add(1, std::memory_order_relaxed);
        g_lastOkFrame = g_frame.load();
        return true;
    }
    if (r == kFailed) g_fallbacks.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// After a forwarded DrawIndexed: the game's TAA resolve ran (cockpit_taa), DLSS for the rest of the image.
void After(ID3D11DeviceContext* c) {
    if (!t_after) return;
    t_after = false;
#ifdef QVFR_HAVE_DLSS
    int r = kNotTaa;
    Protect("after", [&] { r = Evaluate(c, true); });
    if (r == kDone) {
        g_evals.fetch_add(1, std::memory_order_relaxed);
        g_lastOkFrame = g_frame.load();
    } else if (r == kFailed) {
        g_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
#endif
}

void STDMETHODCALLTYPE D_DrawIndexed(ID3D11DeviceContext* c, UINT n, UINT start, INT base) {
    if (Replace(c, n)) return;
    o_drawIndexed(c, n, start, base);
    After(c);
}

bool Inside() { return t_inside; }

void EnsureD3DHooks() {
    if (g_d3dHooked) return;
    ID3D11DeviceContext* imm = gpu::Context();
    if (!imm) return;  // before the first Submit: the game's TAA
    g_d3dHooked = true;
    // Quad mode: sun_fx owns the DrawIndexed hook (a second hook on the same function would fail).
    if (Cfg().quadMode == QuadMode::Quad && sunfx::SetDrawReplace(Replace, Inside, After)) {
        g_external = true;
        Log("DLAA: the TAA resolve draws come through the quad module's DrawIndexed hook (immediate context %p)",
            (void*)imm);
        return;
    }
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
        const bool trace = g_traceLeft.load(std::memory_order_relaxed) > 0 && g_traceLeft.fetch_sub(1) > 0;
        if (g_setLogs < 8 || trace) {  // one frame's worth: which tasks set and clear the offset
            ++g_setLogs;
            Log("DLAA: SetJitter #%d: context %u, (%.4f, %.4f), frame counter %u, from AMS2AVX.exe+0x%llX", g_setLogs, ctx,
                v[0], v[1], counter, (unsigned long long)((uintptr_t)ret - hooks::ExeBase()));
        }
    });
}

bool FrameJitterRead(float out[2]) {
    void* dev = nullptr;
    uint32_t counter = 0;
    if (!g_devGlobal || !hooks::TryRead(g_devGlobal, &dev, sizeof(dev)) || !dev ||
        !hooks::TryRead((const uint8_t*)dev + kFrameCounterOffset, &counter, sizeof(counter)) || counter != g_jitCounter)
        return false;
    out[0] = g_jit[0];
    out[1] = g_jit[1];
    return true;
}

void __fastcall D_Locks(void* taa, void* src, void* normal, uint32_t ctx, void* camera, uint64_t a6, int view) {
    if (Wanted() && Cfg().dlaaSkipLocks && !CockpitTaaNow() && g_evals.load() &&
        g_frame.load() <= g_lastOkFrame.load() + 1) {
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
    if (Cfg().quadMode == QuadMode::Quad && !Cfg().taaViews)
        Log("DLAA: [debug] taa_views=0: the focus views share outer R's TAA targets and get motion vectors without the "
            "focus window's motion (smearing in the focus area)");
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
        if (Pressed(c.dlaaExpKeyMod, c.dlaaExpKey, g_expKeyDown)) {
            g_expStep = (g_expStep.load() + 1) % kExpStepCount;
            Log("DLAA: exposure hotkey: %s%s", ExposureText().c_str(), g_expStep.load() == 0 ? " (the ini settings)" : "");
        }
        if (Pressed(c.dlaaPresetKeyMod, c.dlaaPresetKey, g_presetKeyDown)) {
            g_presetStep = (g_presetStep.load() + 1) % kPresetStepCount;
            Log("DLAA: preset hotkey: %s%s", ExposureText().c_str(), g_presetStep.load() == 0 ? " (the ini preset)" : "");
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
#ifdef QVFR_HAVE_DLSS
    if (g_maskEvals || g_expEvals || g_cockpitEvals)
        Log("DLAA: evaluations with the bias mask %llu, with the fixed exposure %llu, with the cockpit from the game's "
            "TAA %llu (session)",
            (unsigned long long)g_maskEvals, (unsigned long long)g_expEvals, (unsigned long long)g_cockpitEvals);
#endif
    if (Cfg().quadMode != QuadMode::Quad) return;
    mvrmotion::LogStats(totals);
    std::string views;
    for (int v = 1; v < kViews; ++v) {
        const uint64_t n = g_viewEvals[v].load();
        char buf[64];
        snprintf(buf, sizeof(buf), "%s%s %llu", v > 1 ? ", " : "", kViewName[v],
                 (unsigned long long)(totals ? n : n - g_lastViewEvals[v]));
        views += buf;
        g_lastViewEvals[v] = n;
    }
    Log("DLAA%s: DLSS evaluations per view: %s; outside the eye groups %llu in the session",
        totals ? " (session total)" : "", views.c_str(), (unsigned long long)g_viewEvals[0].load());
}

void TraceJitter(int calls) { g_traceLeft.store(calls); }

bool FrameJitter(float out[2]) {
    bool ok = false;
    if (!g_installed || !o_setJitter) return ok;
    Protect("FrameJitter", [&] { ok = FrameJitterRead(out); });
    return ok;
}

void SetShotFrame(bool shot) {
#ifdef QVFR_HAVE_DLSS
    if (shot && !g_shot.load()) g_shotViews = 0;
    g_shot = shot;
#else
    (void)shot;
#endif
}

void BeginGroup(int group) { g_group.store(group >= 0 && group < 4 ? group : -1, std::memory_order_relaxed); }

void EndGroup() { g_group.store(-1, std::memory_order_relaxed); }

std::string Describe() {
    std::lock_guard lock(g_sm);
    return "DLAA: " + g_state;
}

void Shutdown() {
    if (!g_installed) return;
    g_on = false;
#ifdef QVFR_HAVE_DLSS
    Protect("Shutdown", [&] {
        ReleasePending();
        DropNgxLocked();
    });
    g_ngxLogLines = 0;
#endif
    hooks::DisableAll();
}

}  // namespace qvfr::dlaa
