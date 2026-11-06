#include "mvr_pass.h"

#include <d3d11_1.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "clones.h"
#include "common.h"
#include "gpu.h"
#include "hooks.h"
#include "mvr_pass_depth_ps.h"  // kMvrPassDepthPS (fxc from shaders/mvr_pass.hlsl)
#include "mvr_pass_mask_ps.h"   // kMvrPassMaskPS
#include "mvr_pass_vs.h"        // kMvrPassVS
#include "mvr_vs.h"

namespace qvfr::mvrpass {

namespace {

using Microsoft::WRL::ComPtr;

// Engine layout (docs/mvr.md B5, B8), AMS2AVX.exe build aa43bd68.
constexpr uintptr_t kPhase1Vtable = 0x204adc8;  // CPhase1Render (RVA)
constexpr uintptr_t kPhase1Execute = 0x9c96c0;
constexpr uintptr_t kBuildVtable = 0x204ae20;   // CPhase1BuildCommandThread
constexpr uintptr_t kBuildExecute = 0x9c9310;
constexpr uintptr_t kTaskFinish = 0xcde260;     // +0x38 of both
constexpr int kExecuteSlot = 0x48 / 8, kFinishSlot = 0x38 / 8;
constexpr int kVtSlots = 16;
constexpr int kOuterL = 0, kOuterR = 1, kFocusL = 2, kFocusR = 3;
constexpr int kMaxCopies = 10;   // two-slice copies (the G-buffer and what else the MVR part binds)
constexpr int kMaxViews = 64;
constexpr int kMaxTargets = 4;   // outer-size targets of the pass: 3 colors and the depth
constexpr int kMaxMasks = 2;     // targets of the tail (the sun masks)
constexpr int kShotImages = 2 * (kMaxTargets + kMaxMasks);
const char* const kEye[2] = {"L", "R"};

// PartWait: outer's pass began with no target bound (the game binds its G-buffer inside the task); the MVR part starts
// at the first bind or clear.
enum Part { PartNone, PartWait, PartMvr, PartTail };

bool g_installed = false;
std::atomic<bool> g_on{false};
std::atomic<bool> g_dead{false};
uintptr_t g_phase1Vt = 0, g_buildVt = 0;  // absolute vtable addresses
ID3D11DeviceContext* g_testCtx = nullptr;
CloneFn g_cloneFn = nullptr;
JitterFn g_jitterFn = nullptr;
uint32_t g_gen = 0;
uint64_t g_frame = 0;
bool g_enabled = false;  // this frame

ID3D11DeviceContext* Imm() { return g_testCtx ? g_testCtx : gpu::Context(); }

// The focus clone of an engine target (bound as it is: clones::Original gives the engine's texture behind a clone).
ID3D11Texture2D* FocusClone(ID3D11Resource* r) {
    if (!r) return nullptr;
    if (g_cloneFn) return g_cloneFn(r);
    return clones::CloneOf((ID3D11Resource*)clones::Original(r), clones::Focus);
}
bool FocusSwapped() { return g_cloneFn ? true : clones::SwappedIn(clones::Focus); }
bool FrameJitter(float j[2]) { return g_jitterFn && g_jitterFn(j); }

thread_local bool t_pass = false;      // the MVR part of a pass runs on this thread
thread_local bool t_internal = false;  // the module's own D3D11 calls
thread_local uint8_t* t_focusTask = nullptr;  // focus L's CPhase1Render running in a shot frame
thread_local uint8_t* t_fillTask = nullptr;   // overwrite mode: a focus CPhase1Render whose clones get the stash after it
thread_local int t_fillEye = -1;
thread_local bool t_fillDepth = true;

void Fail(const char* where, unsigned long code);

template <class F>
void Protect(const char* where, F&& f) {
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Fail(where, code);
    }
}

struct Internal {
    Internal() { t_internal = true; }
    ~Internal() { t_internal = false; }
};

std::string NvErr(NvAPI_Status s) {
    NvAPI_ShortString msg = {};
    NvAPI_GetErrorMessage(s, msg);
    char buf[96];
    snprintf(buf, sizeof(buf), "%s (%d)", msg, (int)s);
    return buf;
}

D3D11_TEXTURE2D_DESC DescOf(ID3D11Resource* r, bool* ok = nullptr) {
    D3D11_TEXTURE2D_DESC d{};
    ComPtr<ID3D11Texture2D> t;
    const bool got = r && SUCCEEDED(r->QueryInterface(IID_PPV_ARGS(&t)));
    if (got) t->GetDesc(&d);
    if (ok) *ok = got;
    return d;
}

bool SameLayout(const D3D11_TEXTURE2D_DESC& a, const D3D11_TEXTURE2D_DESC& b) {
    return a.Width == b.Width && a.Height == b.Height && a.MipLevels == b.MipLevels && a.ArraySize == b.ArraySize &&
           a.Format == b.Format && a.SampleDesc.Count == b.SampleDesc.Count &&
           a.SampleDesc.Quality == b.SampleDesc.Quality && a.BindFlags == b.BindFlags;
}

// Depth formats: the typeless texture's DSV and SRV formats (unknown: no SRV, the corner cannot be read).
void DepthFormats(DXGI_FORMAT f, DXGI_FORMAT& dsv, DXGI_FORMAT& srv) {
    dsv = srv = DXGI_FORMAT_UNKNOWN;
    switch (f) {
        case DXGI_FORMAT_R32_TYPELESS: dsv = DXGI_FORMAT_D32_FLOAT; srv = DXGI_FORMAT_R32_FLOAT; break;
        case DXGI_FORMAT_R24G8_TYPELESS: dsv = DXGI_FORMAT_D24_UNORM_S8_UINT; srv = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; break;
        case DXGI_FORMAT_R16_TYPELESS: dsv = DXGI_FORMAT_D16_UNORM; srv = DXGI_FORMAT_R16_UNORM; break;
        case DXGI_FORMAT_R32G8X24_TYPELESS:
            dsv = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            srv = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            break;
        default: break;
    }
}

// ---------------------------------------------------------------- the module's own D3D11 objects

struct Gpu {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext1> c1;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> psDepth, psMask;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11DepthStencilState> dssWrite, dssOff;
    ComPtr<ID3D11RasterizerState> rs;
    ComPtr<ID3D11Buffer> rowsCb, maskCb, depthCb;
    uint32_t depthCbData[8] = {};  // in depthCb: bias steps, slope scale (float bits), corner width, height, dilate
    bool failed = false;
};
Gpu g_g;

bool EnsureGpu(ID3D11DeviceContext* c) {
    if (g_g.dev) return true;
    if (g_g.failed || !c) return false;
    Gpu& g = g_g;
    c->GetDevice(&g.dev);
    c->QueryInterface(IID_PPV_ARGS(&g.c1));
    HRESULT hr = g.dev ? S_OK : E_FAIL;
    if (SUCCEEDED(hr)) hr = g.dev->CreateVertexShader(kMvrPassVS, sizeof(kMvrPassVS), nullptr, &g.vs);
    if (SUCCEEDED(hr)) hr = g.dev->CreatePixelShader(kMvrPassDepthPS, sizeof(kMvrPassDepthPS), nullptr, &g.psDepth);
    if (SUCCEEDED(hr)) hr = g.dev->CreatePixelShader(kMvrPassMaskPS, sizeof(kMvrPassMaskPS), nullptr, &g.psMask);
    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (SUCCEEDED(hr)) hr = g.dev->CreateSamplerState(&sd, &g.sampler);
    D3D11_DEPTH_STENCIL_DESC dd{};
    dd.DepthEnable = TRUE;
    dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
    if (SUCCEEDED(hr)) hr = g.dev->CreateDepthStencilState(&dd, &g.dssWrite);
    dd.DepthEnable = FALSE;
    dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    if (SUCCEEDED(hr)) hr = g.dev->CreateDepthStencilState(&dd, &g.dssOff);
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    if (SUCCEEDED(hr)) hr = g.dev->CreateRasterizerState(&rd, &g.rs);
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 8 * 4 * sizeof(float);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_DEFAULT;
    if (SUCCEEDED(hr)) hr = g.dev->CreateBuffer(&bd, nullptr, &g.rowsCb);
    bd.ByteWidth = 8 * sizeof(float);
    if (SUCCEEDED(hr)) hr = g.dev->CreateBuffer(&bd, nullptr, &g.maskCb);
    bd.ByteWidth = 8 * sizeof(uint32_t);
    const uint32_t zero[8] = {};
    const D3D11_SUBRESOURCE_DATA init{zero, 0, 0};
    if (SUCCEEDED(hr)) hr = g.dev->CreateBuffer(&bd, &init, &g.depthCb);
    memset(g.depthCbData, 0, sizeof(g.depthCbData));
    if (FAILED(hr)) {
        Log("MVR G-buffer: ERROR: the module's D3D11 objects could not be made (0x%08lX); off for this device",
            (unsigned long)hr);
        g_g = Gpu{};
        g_g.failed = true;
        return false;
    }
    return true;
}

// The state the module's own passes change (sun_fx's set plus the targets, the viewports and PS slot 0).
struct Saved {
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    D3D11_VIEWPORT vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
    UINT nvp = 0;
    ID3D11InputLayout* il = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY topo = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ID3D11VertexShader* vs = nullptr;
    ID3D11HullShader* hs = nullptr;
    ID3D11DomainShader* ds = nullptr;
    ID3D11GeometryShader* gs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11Buffer* cb = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11SamplerState* sampler = nullptr;
    ID3D11BlendState* bs = nullptr;
    float blendFactor[4] = {};
    UINT sampleMask = 0;
    ID3D11DepthStencilState* dss = nullptr;
    UINT stencilRef = 0;
    ID3D11RasterizerState* rs = nullptr;

    void Save(ID3D11DeviceContext* c) {
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
        nvp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        c->RSGetViewports(&nvp, vp);
        c->IAGetInputLayout(&il);
        c->IAGetPrimitiveTopology(&topo);
        c->VSGetShader(&vs, nullptr, nullptr);
        c->HSGetShader(&hs, nullptr, nullptr);
        c->DSGetShader(&ds, nullptr, nullptr);
        c->GSGetShader(&gs, nullptr, nullptr);
        c->PSGetShader(&ps, nullptr, nullptr);
        c->PSGetConstantBuffers(0, 1, &cb);
        c->PSGetShaderResources(0, 1, &srv);
        c->PSGetSamplers(0, 1, &sampler);
        c->OMGetBlendState(&bs, blendFactor, &sampleMask);
        c->OMGetDepthStencilState(&dss, &stencilRef);
        c->RSGetState(&rs);
    }
    // Inputs first: a saved target that is also the pass's input then binds without a hazard.
    void Restore(ID3D11DeviceContext* c) {
        c->PSSetShaderResources(0, 1, &srv);
        c->PSSetSamplers(0, 1, &sampler);
        c->PSSetConstantBuffers(0, 1, &cb);
        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
        c->RSSetViewports(nvp, vp);
        c->IASetInputLayout(il);
        c->IASetPrimitiveTopology(topo);
        c->VSSetShader(vs, nullptr, 0);
        c->HSSetShader(hs, nullptr, 0);
        c->DSSetShader(ds, nullptr, 0);
        c->GSSetShader(gs, nullptr, 0);
        c->PSSetShader(ps, nullptr, 0);
        c->OMSetBlendState(bs, blendFactor, sampleMask);
        c->OMSetDepthStencilState(dss, stencilRef);
        c->RSSetState(rs);
        auto rel = [](auto*& p) {
            if (p) p->Release();
            p = nullptr;
        };
        for (auto*& v : rtv) rel(v);
        rel(dsv);
        rel(il);
        rel(vs);
        rel(hs);
        rel(ds);
        rel(gs);
        rel(ps);
        rel(cb);
        rel(srv);
        rel(sampler);
        rel(bs);
        rel(dss);
        rel(rs);
    }
};

// One full-screen triangle with the module's VS into the bound targets over w x h.
void FullScreen(ID3D11DeviceContext* c, ID3D11PixelShader* ps, ID3D11DepthStencilState* dss, UINT w, UINT h) {
    const D3D11_VIEWPORT vp{0, 0, (float)w, (float)h, 0, 1};
    c->RSSetViewports(1, &vp);
    c->IASetInputLayout(nullptr);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(g_g.vs.Get(), nullptr, 0);
    c->HSSetShader(nullptr, nullptr, 0);
    c->DSSetShader(nullptr, nullptr, 0);
    c->GSSetShader(nullptr, nullptr, 0);
    c->PSSetShader(ps, nullptr, 0);
    c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    c->OMSetDepthStencilState(dss, 0);
    c->RSSetState(g_g.rs.Get());
    c->Draw(3, 0);
}

// The depth of slice 1's corner (w x h) into `dsv`.
void DepthPass(ID3D11DeviceContext* c, ID3D11ShaderResourceView* src, ID3D11DepthStencilView* dsv, UINT w, UINT h,
               uint32_t biasUlps, float slope, bool dilate) {
    Internal in;
    uint32_t v[8] = {biasUlps, 0, w, h, dilate ? 1u : 0u, 0, 0, 0};
    memcpy(&v[1], &slope, sizeof(slope));
    if (memcmp(v, g_g.depthCbData, sizeof(v)) != 0) {
        c->UpdateSubresource(g_g.depthCb.Get(), 0, nullptr, v, 0, 0);
        memcpy(g_g.depthCbData, v, sizeof(v));
    }
    Saved s;
    s.Save(c);
    ID3D11Buffer* cb1 = nullptr;
    c->PSGetConstantBuffers(1, 1, &cb1);
    c->OMSetRenderTargets(0, nullptr, dsv);
    c->PSSetShaderResources(0, 1, &src);
    ID3D11Buffer* cb = g_g.depthCb.Get();
    c->PSSetConstantBuffers(1, 1, &cb);
    FullScreen(c, g_g.psDepth.Get(), g_g.dssWrite.Get(), w, h);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    c->PSSetConstantBuffers(1, 1, &cb1);
    if (cb1) cb1->Release();
    s.Restore(c);
}

// A mask of outer read in the focus window into `rtv` (w x h): outer UV = origin + focus UV * scale.
void MaskPass(ID3D11DeviceContext* c, ID3D11ShaderResourceView* src, ID3D11RenderTargetView* rtv, UINT w, UINT h,
              const float origin[2], const float scale[2]) {
    Internal in;
    const float k[8] = {origin[0], origin[1], scale[0], scale[1], 1.0f / w, 1.0f / h, 0, 0};
    c->UpdateSubresource(g_g.maskCb.Get(), 0, nullptr, k, 0, 0);
    Saved s;
    s.Save(c);
    c->OMSetRenderTargets(1, &rtv, nullptr);
    c->PSSetShaderResources(0, 1, &src);
    ID3D11SamplerState* smp = g_g.sampler.Get();
    c->PSSetSamplers(0, 1, &smp);
    ID3D11Buffer* cb = g_g.maskCb.Get();
    c->PSSetConstantBuffers(0, 1, &cb);
    FullScreen(c, g_g.psMask.Get(), g_g.dssOff.Get(), w, h);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    s.Restore(c);
}

// ---------------------------------------------------------------- two-slice copies of the pass's targets

struct Copy {
    ID3D11Resource* orig = nullptr;  // the game's texture as bound (only compared)
    ComPtr<ID3D11Texture2D> arr;
    D3D11_TEXTURE2D_DESC desc{};     // the game's
    ComPtr<ID3D11ShaderResourceView> depthSrv;  // depth: both slices
    bool depth = false;
    bool used = false;               // in the current pass
    uint64_t lastFrame = 0;
};
struct ViewMap {
    void* orig = nullptr;  // the game's view (only compared)
    int copy = -1;
    ComPtr<ID3D11View> arr;
};
std::vector<Copy> g_copies;
std::vector<ViewMap> g_views;

void DropViewsOf(int k) {
    g_views.erase(std::remove_if(g_views.begin(), g_views.end(), [&](const ViewMap& m) { return m.copy == k; }),
                  g_views.end());
}

// ---------------------------------------------------------------- the pass and the stash of each eye

struct Pass {
    Part part = PartNone;
    int eye = -1;
    uint8_t* task = nullptr;
    uint32_t w = 0, h = 0, fw = 0, fh = 0;
    bool mFilled = false, jitterKnown = false;
    int jitterTries = 0;
    float jit[2] = {};
    float rows[8][4] = {};
    NvU32 views = 1;
    ID3D11Buffer* slot3 = nullptr;  // the game's binding of VS slot 3 (referenced)
    UINT slot3First = 0, slot3Num = 0;
    std::vector<std::pair<ID3D11Resource*, DXGI_FORMAT>> masks;  // tail targets (only compared)
    unsigned binds = 0, clears = 0, mapFailures = 0, mixed = 0, patchedSets = 0, plainSets = 0, cbRebinds = 0;
    unsigned cbGame = 0, viewsFailures = 0, lateBinds = 0, tailBinds = 0, copiedBack = 0, draws = 0, plainDraws = 0;
    unsigned lateStarts = 0;  // 1: the MVR part started at the task's first bind or clear (not at its start)
    unsigned waitBinds = 0;   // binds and clears of other sizes before it
    std::string tailSeq;      // the tail's binds and draws (logged for the first passes with late binds)
    unsigned tailDraws = 0;   // draws since the latest tail bind
    bool lateBound = false;   // the latest tail bind holds an outer-size target
    unsigned lateDraws = 0;   // draws with an outer-size target bound in the tail (they would miss focus)
    bool failed = false;
    std::string why;
    double t0 = 0;
};
Pass g_p;

struct StashG {
    ID3D11Resource* orig = nullptr;  // the game's target as bound (only compared)
    ComPtr<ID3D11Texture2D> tex;     // the focus clone's layout
    D3D11_TEXTURE2D_DESC desc{};
    ComPtr<ID3D11DepthStencilView> dsv;
    bool depth = false;
};
struct StashM {
    ID3D11Resource* orig = nullptr;
    ComPtr<ID3D11Texture2D> tex;  // outer's mask
    D3D11_TEXTURE2D_DESC desc{};
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;  // the game's RTV format
    ComPtr<ID3D11ShaderResourceView> srv;
    ID3D11Resource* rtvFor = nullptr;  // focus target of `rtv` (only compared)
    ComPtr<ID3D11RenderTargetView> rtv;
};
struct Stash {
    StashG g[kMaxTargets];
    int n = 0;
    StashM m[kMaxMasks];
    int nm = 0;
    bool valid = false;
    uint64_t frame = 0;
    double sx = 1, ox = 0, sy = 1, oy = 0;
};
Stash g_stash[2];

struct Win {
    bool valid = false;
    uint64_t frame = 0;
    double sx = 1, ox = 0, sy = 1, oy = 0;
    uint32_t w = 0, h = 0;  // outer's target size (the game's render size)
};
Win g_win[2];
struct FocusState {
    bool projected = false;
    uint64_t frame = 0;
    bool decided = false, skip = false;
};
FocusState g_focus[2];

// Counters per stats interval and for the session.
struct Reason {
    std::string text;
    uint64_t n = 0;
};
struct Stats {
    uint64_t frames = 0, passes[2] = {}, passFailed[2] = {}, skipped[2] = {}, own[2] = {}, fills = 0;
    uint64_t fillFailures = 0, masks = 0, jitterKnown = 0, jitterUnknown = 0, plainSets = 0, cbRebinds = 0;
    uint64_t cbGame = 0, mixed = 0, lateBinds = 0, mapFailures = 0, viewsFailures = 0;
    uint64_t lateStarts = 0, waitBinds = 0, lateDraws = 0;
    uint64_t switches[2] = {};  // the focus group's choice (from outer / own pass) differs from the previous frame's
    uint64_t overwrites[2] = {};  // overwrite mode: own pass run, then the stash on top
    double passMs[2] = {};
    std::vector<Reason> reasons;
};
Stats g_st, g_tot;
std::string g_lastFail;  // the latest pass failure (logged once per text)
int g_tailLogs = 0;      // tails with late binds logged in this session
Mode g_mode = Mode::Skip;  // the diagnosis hotkey (mvr_pass.h)
int g_testBias = -1;  // smoke: overrides the settings when >= 0
float g_testSlope = 0;
bool g_testDilate = false;
// The conservative depth (mvr_pass.hlsl, PSDepth) in every mode but SkipRaw.
uint32_t DepthBias() {
    if (g_mode == Mode::SkipRaw) return 0u;
    return (uint32_t)(g_testBias >= 0 ? g_testBias : Cfg().mvrDepthBias);
}
float DepthSlope() {
    if (g_mode == Mode::SkipRaw) return 0.0f;
    return g_testBias >= 0 ? g_testSlope : Cfg().mvrDepthSlope;
}
bool DepthDilate() {
    if (g_mode == Mode::SkipRaw) return false;
    return g_testBias >= 0 ? g_testDilate : Cfg().mvrDepthDilate;
}
int g_lastSkip[2] = {-1, -1};  // the focus group's choice in its latest frame (-1: none yet)

void AddReason(std::vector<Reason>& v, const std::string& why) {
    for (Reason& r : v)
        if (r.text == why) {
            ++r.n;
            return;
        }
    if (v.size() < 16) v.push_back({why, 1});
}
void Own(int eye, const std::string& why) {
    ++g_st.own[eye];
    ++g_tot.own[eye];
    char b[160];
    snprintf(b, sizeof(b), "%s %s", kEye[eye], why.c_str());
    AddReason(g_st.reasons, b);
    AddReason(g_tot.reasons, b);
}

// The shot comparison (eye L): focus L's own pass vs the stash.
struct Shot {
    bool armed = false, reported = false;
    int base = -1;
    uint64_t frame = 0;
    bool outerDone = false, focusBefore = false, focusDone = false;
    Pass pass;  // outer L's (counters and rows; no references)
    int n = 0, nm = 0;
    ComPtr<ID3D11Texture2D> mask[kMaxMasks];  // ours at the focus mask's size
    std::vector<std::pair<std::string, unsigned>> plain;  // unpatched VS sets by DXBC checksum
};
Shot g_shot;
// The shot's image names and pair labels: the shots module reads them after the frame (the PNGs and comparisons run
// later), so they live apart from g_shot, which the next frame start resets.
char g_shotLabels[kMaxTargets + kMaxMasks][96] = {};
char g_shotNames[kShotImages][32] = {};

// Test (smoke) snapshot of the latest pass.
TestState g_test;

bool SetViews(ID3D11DeviceContext* c, NvU32 views) {
    NV_MULTIVIEW_PARAMS p = {};
    p.version = NV_MULTIVIEW_PARAMS_VER;
    p.numViews = views;
    for (NvU32 i = 0; i < 4; ++i) p.renderTargetIndexOffset[i] = i;
    const NvAPI_Status st = NvAPI_D3D_SetMultiViewMode(c, &p);
    if (st == NVAPI_OK) return true;
    if (!g_p.viewsFailures++ && !g_st.viewsFailures)
        Log("MVR G-buffer: NvAPI_D3D_SetMultiViewMode(%u views) %s", (unsigned)views, NvErr(st).c_str());
    return false;
}

void ReleasePassRefs(Pass& p) {
    if (p.slot3) p.slot3->Release();
    p.slot3 = nullptr;
}

void ReleaseAll() {
    ReleasePassRefs(g_p);
    g_p = Pass{};
    g_copies.clear();
    g_views.clear();
    for (Stash& s : g_stash) s = Stash{};
    g_shot = Shot{};
    g_g = Gpu{};
}

void Fail(const char* where, unsigned long code) {
    g_on = false;
    if (t_pass && Imm()) SetViews(Imm(), 1);
    t_pass = false;
    t_internal = false;
    g_test.faulted = true;
    if (g_dead.exchange(true)) return;
    Log("MVR G-buffer: ERROR: exception 0x%08lX in %s; off for this session (the focus groups run their own pass)",
        code, where);
}

// ---------------------------------------------------------------- copies and views

int CopyFor(ID3D11DeviceContext* c, ID3D11Resource* res) {
    bool ok = false;
    const D3D11_TEXTURE2D_DESC d = DescOf(res, &ok);
    if (!ok || d.ArraySize != 1) return -1;
    int slot = -1;
    for (int k = 0; k < (int)g_copies.size(); ++k) {
        Copy& cp = g_copies[k];
        if (cp.orig != res) continue;
        if (SameLayout(cp.desc, d)) {
            cp.used = true;
            cp.lastFrame = g_frame;
            return k;
        }
        slot = k;  // a new texture at the address of a released one
        break;
    }
    if (slot < 0 && (int)g_copies.size() >= kMaxCopies) {
        for (int k = 0; k < (int)g_copies.size() && slot < 0; ++k)
            if (g_copies[k].lastFrame + 2 < g_frame) slot = k;
        if (slot < 0) return -1;
    }
    Copy cp;
    cp.orig = res;
    cp.desc = d;
    cp.depth = (d.BindFlags & D3D11_BIND_DEPTH_STENCIL) != 0;
    D3D11_TEXTURE2D_DESC a = d;
    a.ArraySize = 2;
    a.MiscFlags &= ~(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                     D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE |
                     D3D11_RESOURCE_MISC_TEXTURECUBE);
    DXGI_FORMAT dsvF, srvF;
    DepthFormats(d.Format, dsvF, srvF);
    if (cp.depth && srvF != DXGI_FORMAT_UNKNOWN) a.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    if (!dev || FAILED(dev->CreateTexture2D(&a, nullptr, &cp.arr))) return -1;
    if (cp.depth && srvF != DXGI_FORMAT_UNKNOWN) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = srvF;
        if (d.SampleDesc.Count > 1) {
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
            sv.Texture2DMSArray = {0, 2};
        } else {
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            sv.Texture2DArray = {0, 1, 0, 2};
        }
        dev->CreateShaderResourceView(cp.arr.Get(), &sv, &cp.depthSrv);
    }
    cp.used = true;
    cp.lastFrame = g_frame;
    if (slot >= 0) {
        DropViewsOf(slot);
        g_copies[slot] = std::move(cp);
        return slot;
    }
    g_copies.push_back(std::move(cp));
    return (int)g_copies.size() - 1;
}

// The copy's view for a view of the game (checked against its resource: views are released and made again); a view
// of a copy itself (the pass read the bound views back) stays.
ID3D11View* FindView(void* v, int* copy = nullptr) {
    for (size_t i = 0; i < g_views.size(); ++i) {
        ViewMap& m = g_views[i];
        if (m.arr.Get() == v) {
            if (copy) *copy = m.copy;
            return m.arr.Get();
        }
        if (m.orig != v) continue;
        ComPtr<ID3D11Resource> res;
        static_cast<ID3D11View*>(v)->GetResource(&res);
        if (m.copy >= 0 && m.copy < (int)g_copies.size() && res.Get() == g_copies[m.copy].orig &&
            SameLayout(DescOf(res.Get()), g_copies[m.copy].desc)) {
            g_copies[m.copy].used = true;
            g_copies[m.copy].lastFrame = g_frame;
            if (copy) *copy = m.copy;
            return m.arr.Get();
        }
        g_views.erase(g_views.begin() + i);
        return nullptr;
    }
    return nullptr;
}

void AddView(void* v, int k, ID3D11View* arr) {
    if (g_views.size() >= kMaxViews)
        g_views.erase(std::remove_if(g_views.begin(), g_views.end(),
                                     [](const ViewMap& m) {
                                         return m.copy < 0 || m.copy >= (int)g_copies.size() ||
                                                g_copies[m.copy].lastFrame < g_frame;
                                     }),
                      g_views.end());
    ViewMap m;
    m.orig = v;
    m.copy = k;
    m.arr = arr;
    g_views.push_back(std::move(m));
}

// The copy's view on both slices for a view of the game, or null (counted).
ID3D11RenderTargetView* ArrayRtv(ID3D11DeviceContext* c, ID3D11RenderTargetView* v) {
    if (ID3D11View* m = FindView(v)) return static_cast<ID3D11RenderTargetView*>(m);
    ComPtr<ID3D11Resource> res;
    v->GetResource(&res);
    const int k = CopyFor(c, res.Get());
    D3D11_RENDER_TARGET_VIEW_DESC d{};
    v->GetDesc(&d);
    bool ok = k >= 0;
    if (d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D) {
        const UINT mip = d.Texture2D.MipSlice;
        d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {mip, 0, 2};
    } else if (d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DMS) {
        d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, 2};
    } else {
        ok = false;
    }
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    ComPtr<ID3D11RenderTargetView> out;
    if (!ok || !dev || FAILED(dev->CreateRenderTargetView(g_copies[k].arr.Get(), &d, &out))) {
        ++g_p.mapFailures;
        return nullptr;
    }
    AddView(v, k, out.Get());
    return out.Get();
}

ID3D11DepthStencilView* ArrayDsv(ID3D11DeviceContext* c, ID3D11DepthStencilView* v) {
    if (ID3D11View* m = FindView(v)) return static_cast<ID3D11DepthStencilView*>(m);
    ComPtr<ID3D11Resource> res;
    v->GetResource(&res);
    const int k = CopyFor(c, res.Get());
    D3D11_DEPTH_STENCIL_VIEW_DESC d{};
    v->GetDesc(&d);
    bool ok = k >= 0;
    if (d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2D) {
        const UINT mip = d.Texture2D.MipSlice;
        d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {mip, 0, 2};
    } else if (d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DMS) {
        d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, 2};
    } else {
        ok = false;
    }
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    ComPtr<ID3D11DepthStencilView> out;
    if (!ok || !dev || FAILED(dev->CreateDepthStencilView(g_copies[k].arr.Get(), &d, &out))) {
        ++g_p.mapFailures;
        return nullptr;
    }
    AddView(v, k, out.Get());
    return out.Get();
}

// The game's view behind a bound view of a copy (null for null; a view that is not a copy's stays).
void* OrigView(ID3D11View* v) {
    if (!v) return nullptr;
    for (const ViewMap& m : g_views)
        if (m.arr.Get() == v) return m.orig;
    return v;
}

// True when the view's texture has the pass's outer size.
bool OuterSize(ID3D11View* v) {
    int k = -1;
    if (FindView(v, &k) && k >= 0) return g_copies[k].desc.Width == g_p.w && g_copies[k].desc.Height == g_p.h;
    ComPtr<ID3D11Resource> res;
    v->GetResource(&res);
    const D3D11_TEXTURE2D_DESC d = DescOf(res.Get());
    return d.Width == g_p.w && d.Height == g_p.h;
}

// ---------------------------------------------------------------- view 1

// View 1 = the eye's focus in the top-left corner of slice 1 (a = fw / w, b = fh / h of the outer size). From outer's
// clip space: focus x_f = sx x + ox w, y_f = sy y + oy w (NDC y up); corner x_c = a x_f + (a - 1) w,
// y_c = b y_f + (1 - b) w, so focus pixel (i, j) lands on pixel (i, j) of the slice. The clip planes keep
// -w <= x_f, y_f <= w (the window, in view 0's clip space). TAA jitter: outer's clip position carries outer's (the
// engine's offset j over the render size, NDC = j / w_px), the focus group's own carries j / fw; x_f then gets
// ox - sx jx / w + jx / fw (and y the same with h and fh).
void BuildRows(Pass& p, const Win& win) {
    double ox = win.ox, oy = win.oy;
    if (p.jitterKnown) {
        ox += -win.sx * p.jit[0] / p.w + p.jit[0] / p.fw;
        oy += -win.sy * p.jit[1] / p.h + p.jit[1] / p.fh;
    }
    const double a = (double)p.fw / p.w, b = (double)p.fh / p.h, sx = win.sx, sy = win.sy;
    const float rows[8][4] = {
        {(float)(a * sx), 0, 0, (float)(a * ox + a - 1)},
        {0, (float)(b * sy), 0, (float)(b * oy + 1 - b)},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
        {(float)sx, 0, 0, (float)(ox + 1)},
        {(float)-sx, 0, 0, (float)(1 - ox)},
        {0, (float)sy, 0, (float)(oy + 1)},
        {0, (float)-sy, 0, (float)(1 - oy)},
    };
    memcpy(p.rows, rows, sizeof(rows));
}

// At the pass's patched VS sets until this frame's jitter is known (the engine sets it while it records the pass,
// before the first replayed draw).
void FillM(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    const bool want = Cfg().mvrJitterFix;
    if (p.mFilled && (!want || p.jitterKnown || p.jitterTries >= 64)) return;
    float j[2] = {};
    const bool known = want && (++p.jitterTries, FrameJitter(j));
    if (p.mFilled && !known) return;
    p.jitterKnown = known;
    if (known) memcpy(p.jit, j, sizeof(j));
    BuildRows(p, g_win[p.eye]);
    Internal in;
    c->UpdateSubresource(g_g.rowsCb.Get(), 0, nullptr, p.rows, 0, 0);
    p.mFilled = true;
}

// ---------------------------------------------------------------- outer's pass

void PassFailed(Pass& p, const char* why) {
    if (!p.failed) p.why = why;
    p.failed = true;
}

// Slot 3 back to the game's binding.
void RestoreSlot3(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    Internal in;
    if (g_g.c1 && p.slot3 && p.slot3Num)
        g_g.c1->VSSetConstantBuffers1(mvrvs::kCbSlot, 1, &p.slot3, &p.slot3First, &p.slot3Num);
    else
        c->VSSetConstantBuffers(mvrvs::kCbSlot, 1, &p.slot3);
}

// The corners of slice 1 of the outer-size copies into the eye's stash.
void Extract(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    Stash& s = g_stash[p.eye];
    s.n = 0;
    for (Copy& cp : g_copies) {
        if (!cp.used || cp.desc.Width != p.w || cp.desc.Height != p.h) continue;
        if (s.n >= kMaxTargets) return PassFailed(p, "more than 4 outer-size targets");
        if (cp.desc.SampleDesc.Count > 1) return PassFailed(p, "a multisampled target");
        ID3D11Texture2D* clone = FocusClone(cp.orig);
        if (!clone) return PassFailed(p, "a target without a focus clone");
        D3D11_TEXTURE2D_DESC cd{};
        clone->GetDesc(&cd);
        if (cd.Width != p.fw || cd.Height != p.fh || cd.Format != cp.desc.Format || cd.ArraySize != 1 ||
            cd.SampleDesc.Count != 1)
            return PassFailed(p, "a focus clone of another size or format");
        StashG& g = s.g[s.n];
        if (!g.tex || !SameLayout(g.desc, cd)) {
            g = StashG{};
            D3D11_TEXTURE2D_DESC sd = cd;
            sd.MiscFlags &= ~(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                              D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE);
            if (FAILED(g_g.dev->CreateTexture2D(&sd, nullptr, &g.tex))) return PassFailed(p, "stash texture");
            g.desc = cd;
            g.depth = cp.depth;
            if (g.depth) {
                DXGI_FORMAT dsvF, srvF;
                DepthFormats(cd.Format, dsvF, srvF);
                D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
                dv.Format = dsvF;
                dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
                if (dsvF == DXGI_FORMAT_UNKNOWN || FAILED(g_g.dev->CreateDepthStencilView(g.tex.Get(), &dv, &g.dsv))) {
                    g = StashG{};
                    return PassFailed(p, "stash depth view");
                }
            }
        }
        g.orig = cp.orig;
        if (g.depth) {
            if (!cp.depthSrv) return PassFailed(p, "the depth format has no shader view");
            DepthPass(c, cp.depthSrv.Get(), g.dsv.Get(), p.fw, p.fh, DepthBias(), DepthSlope(), DepthDilate());
        } else {
            const D3D11_BOX box{0, 0, 0, p.fw, p.fh, 1};
            Internal in;
            c->CopySubresourceRegion(g.tex.Get(), 0, 0, 0, 0, cp.arr.Get(),
                                     D3D11CalcSubresource(0, 1, cp.desc.MipLevels), &box);
        }
        ++s.n;
    }
    if (!s.n) PassFailed(p, "no outer-size target");
}

// End of the MVR part: MVR off, slot 3 back, the game's views bound again (rebind; at a tail bind the game binds its
// own next), slice 0 into the game's targets, slice 1's corners into the stash. The rest of the task runs plain.
void EndMvr(ID3D11DeviceContext* c, bool rebind) {
    Pass& p = g_p;
    t_pass = false;
    if (p.views != 1) SetViews(c, 1);
    p.views = 1;
    RestoreSlot3(c);
    if (rebind) {
        ID3D11RenderTargetView* rt[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        ID3D11DepthStencilView* ds = nullptr;
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, &ds);
        ID3D11RenderTargetView* back[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        for (int i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
            back[i] = static_cast<ID3D11RenderTargetView*>(OrigView(rt[i]));
        {
            Internal in;
            c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, back,
                                  static_cast<ID3D11DepthStencilView*>(OrigView(ds)));
        }
        for (ID3D11RenderTargetView* v : rt)
            if (v) v->Release();
        if (ds) ds->Release();
    }
    {
        Internal in;
        for (Copy& cp : g_copies) {
            if (!cp.used) continue;
            for (UINT m = 0; m < cp.desc.MipLevels; ++m)
                c->CopySubresourceRegion(cp.orig, D3D11CalcSubresource(m, 0, cp.desc.MipLevels), 0, 0, 0, cp.arr.Get(),
                                         D3D11CalcSubresource(m, 0, cp.desc.MipLevels), nullptr);
            ++p.copiedBack;
        }
    }
    p.part = PartTail;
    if (p.mapFailures) PassFailed(p, "a target could not be mapped to a copy");
    if (p.viewsFailures) PassFailed(p, "SetMultiViewMode failed");
    if (p.cbGame) PassFailed(p, "the game bound VS slot 3");
    if (!p.failed) Extract(c);
}

// A bind in the tail: its targets are the masks (an outer-size one again is drawn without view 1: no stash).
// "WxH fN" of a view's texture (and its view format), for the tail log.
std::string ViewText(ID3D11View* v) {
    if (!v) return "-";
    ComPtr<ID3D11Resource> res;
    v->GetResource(&res);
    bool ok = false;
    const D3D11_TEXTURE2D_DESC d = DescOf(res.Get(), &ok);
    if (!ok) return "?";
    char b[64];
    snprintf(b, sizeof(b), "%ux%u f%d%s", d.Width, d.Height, (int)d.Format, d.SampleDesc.Count > 1 ? " ms" : "");
    return b;
}

void NoteTail(UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv) {
    Pass& p = g_p;
    ++p.tailBinds;
    if (p.tailSeq.size() < 1500) {
        std::string t = p.tailSeq.empty() ? "" : " (" + std::to_string(p.tailDraws) + " draws); ";
        t += "rt";
        for (UINT i = 0; i < n && rtv; ++i)
            if (rtv[i]) t += " " + ViewText(rtv[i]);
        t += " ds " + ViewText(dsv);
        p.tailSeq += t;
    }
    p.tailDraws = 0;
    p.lateBound = false;
    if (dsv && OuterSize(dsv)) {
        ++p.lateBinds;
        p.lateBound = true;
    }
    for (UINT i = 0; i < n && rtv; ++i) {
        if (!rtv[i]) continue;
        if (OuterSize(rtv[i])) {
            ++p.lateBinds;
            p.lateBound = true;
            continue;
        }
        ComPtr<ID3D11Resource> res;
        rtv[i]->GetResource(&res);
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        rtv[i]->GetDesc(&d);
        if (d.ViewDimension != D3D11_RTV_DIMENSION_TEXTURE2D || d.Texture2D.MipSlice != 0) continue;
        bool seen = false;
        for (const auto& m : p.masks) seen |= m.first == res.Get();
        if (!seen && p.masks.size() < kMaxMasks) p.masks.push_back({res.Get(), d.Format});
    }
}

// The resource behind the depth view, else the first color view (null: nothing).
ComPtr<ID3D11Resource> MainOf(UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv) {
    ComPtr<ID3D11Resource> r;
    if (dsv) {
        dsv->GetResource(&r);
        return r;
    }
    for (UINT i = 0; i < n && rtv; ++i)
        if (rtv[i]) {
            rtv[i]->GetResource(&r);
            break;
        }
    return r;
}

// The MVR part of outer's pass starts on `main` (the depth of the first targets): null, or why the pass runs plain.
// The caller binds the targets after it, through the detour.
const char* StartMvr(ID3D11DeviceContext* c, ID3D11Resource* main, int e, uint8_t* task, double t0) {
    Pass& p = g_p;
    const Win& win = g_win[e];
    bool ok = false;
    const D3D11_TEXTURE2D_DESC md = DescOf(main, &ok);
    ID3D11Texture2D* clone = ok ? FocusClone(main) : nullptr;
    D3D11_TEXTURE2D_DESC cd{};
    if (clone) clone->GetDesc(&cd);
    const char* why = !ok ? "no 2D target in outer's pass"
                      : win.w && (md.Width != win.w || md.Height != win.h)
                          ? "the first target of outer's pass is not at the game's size"
                      : md.ArraySize != 1 || md.SampleDesc.Count != 1 ? "outer's depth is an array or multisampled"
                      : !clone                                         ? "no focus clone of outer's depth"
                      : cd.Width > md.Width || cd.Height > md.Height   ? "focus clones larger than outer"
                                                                       : nullptr;
    if (why) return why;
    p = Pass{};
    p.part = PartMvr;
    p.eye = e;
    p.task = task;
    p.w = md.Width;
    p.h = md.Height;
    p.fw = cd.Width;
    p.fh = cd.Height;
    p.t0 = t0;
    for (Copy& cp : g_copies) cp.used = false;
    {
        Internal in;
        if (g_g.c1)
            g_g.c1->VSGetConstantBuffers1(mvrvs::kCbSlot, 1, &p.slot3, &p.slot3First, &p.slot3Num);
        else
            c->VSGetConstantBuffers(mvrvs::kCbSlot, 1, &p.slot3);
        ID3D11Buffer* cb = g_g.rowsCb.Get();
        c->VSSetConstantBuffers(mvrvs::kCbSlot, 1, &cb);
    }
    ID3D11VertexShader* vs = nullptr;
    c->VSGetShader(&vs, nullptr, nullptr);
    const bool patched = mvrvs::IsPatched(vs);
    if (vs) vs->Release();
    FillM(c);
    if (patched && SetViews(c, 2)) p.views = 2;
    t_pass = true;
    return nullptr;
}

void PassBegin(uint8_t* task, int e) {
    Pass& p = g_p;
    Stash& s = g_stash[e];
    s.valid = false;
    if (p.part != PartNone) {  // a pass whose task did not end (should not happen)
        ReleasePassRefs(p);
        p = Pass{};
    }
    if (!g_enabled) return;
    const Win& win = g_win[e];
    if (!win.valid || win.frame != g_frame) return Own(e, "no focus window in outer's pass");
    ID3D11DeviceContext* c = Imm();
    if (!c || !EnsureGpu(c)) return Own(e, "no D3D11 objects");
    ID3D11RenderTargetView* rt[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* ds = nullptr;
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, &ds);
    auto release = [&] {
        for (ID3D11RenderTargetView* v : rt)
            if (v) v->Release();
        if (ds) ds->Release();
    };
    const ComPtr<ID3D11Resource> main = MainOf(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, ds);
    if (!main) {  // the game binds the G-buffer inside the task
        release();
        p = Pass{};
        p.part = PartWait;
        p.eye = e;
        p.task = task;
        p.t0 = NowMs();
        return;
    }
    if (const char* why = StartMvr(c, main.Get(), e, task, NowMs())) {
        release();
        return Own(e, why);
    }
    // The targets the previous tasks left: bound again through the detour, so the pass starts on copies.
    c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rt, ds);
    release();
}

// A bind or clear while outer's pass waits for its first targets: the MVR part starts on them (true) or the pass runs
// plain (false, the reason counted).
bool StartAtFirst(ID3D11DeviceContext* c, ID3D11Resource* main) {
    Pass& p = g_p;
    const int e = p.eye;
    const Win& win = g_win[e];
    bool ok = false;
    const D3D11_TEXTURE2D_DESC md = DescOf(main, &ok);
    if (ok && win.w && (md.Width != win.w || md.Height != win.h)) {  // not the G-buffer yet: wait on
        ++p.waitBinds;
        return false;
    }
    const unsigned waited = p.waitBinds;
    if (const char* why = StartMvr(c, main, e, p.task, p.t0)) {
        p = Pass{};
        Own(e, why);
        return false;
    }
    g_p.lateStarts = 1;
    g_p.waitBinds = waited;
    return true;
}

void ShotOuter(ID3D11DeviceContext* c);
void ShotFocusBefore(ID3D11DeviceContext* c);
void ShotFocusAfter(ID3D11DeviceContext* c);
void ReportShot();

void PassEnd(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    if (p.part == PartWait) {  // the task bound nothing
        const int e = p.eye;
        p = Pass{};
        return Own(e, "outer's pass bound no target at the game's size");
    }
    if (p.part == PartMvr) EndMvr(c, true);
    Stash& s = g_stash[p.eye];
    s.nm = 0;
    // The game binds the G-buffer again after the masks without drawing (run 3): only draws there miss focus.
    if (p.lateDraws) {
        PassFailed(p, "draws into an outer-size target after the masks");
        if (g_tailLogs < 4) {
            ++g_tailLogs;
            Log("MVR G-buffer: outer %s's tail in frame %llu (late binds %u, late draws %u): %s (%u draws)", kEye[p.eye],
                (unsigned long long)g_frame, p.lateBinds, p.lateDraws, p.tailSeq.c_str(), p.tailDraws);
        }
    }
    for (size_t i = 0; i < p.masks.size() && !p.failed; ++i) {
        StashM& m = s.m[s.nm];
        ID3D11Resource* res = p.masks[i].first;
        bool ok = false;
        const D3D11_TEXTURE2D_DESC d = DescOf(res, &ok);
        if (!ok || d.ArraySize != 1 || d.SampleDesc.Count != 1) continue;
        if (!m.tex || !SameLayout(m.desc, d) || m.fmt != p.masks[i].second) {
            m = StashM{};
            D3D11_TEXTURE2D_DESC sd = d;
            sd.MiscFlags = 0;
            sd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
            sv.Format = p.masks[i].second;
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sv.Texture2D = {0, 1};
            if (FAILED(g_g.dev->CreateTexture2D(&sd, nullptr, &m.tex)) ||
                FAILED(g_g.dev->CreateShaderResourceView(m.tex.Get(), &sv, &m.srv))) {
                m = StashM{};
                PassFailed(p, "mask stash");
                break;
            }
            m.desc = d;
            m.fmt = p.masks[i].second;
        }
        m.orig = res;
        {
            Internal in;
            c->CopyResource(m.tex.Get(), res);
        }
        ++s.nm;
    }
    s.valid = !p.failed && s.n > 0;
    s.frame = g_frame;
    const Win& win = g_win[p.eye];
    s.sx = win.sx;
    s.ox = win.ox;
    s.sy = win.sy;
    s.oy = win.oy;
    const int e = p.eye;
    ++g_st.passes[e];
    ++g_tot.passes[e];
    const double ms = NowMs() - p.t0;
    g_st.passMs[e] += ms;
    g_tot.passMs[e] += ms;
    if (p.failed) {
        ++g_st.passFailed[e];
        ++g_tot.passFailed[e];
        if (p.why != g_lastFail) {
            g_lastFail = p.why;
            Log("MVR G-buffer: outer %s's pass gave no stash in frame %llu: %s (logged once per reason)", kEye[e],
                (unsigned long long)g_frame, p.why.c_str());
        }
    }
    (p.jitterKnown ? g_st.jitterKnown : g_st.jitterUnknown) += 1;
    (p.jitterKnown ? g_tot.jitterKnown : g_tot.jitterUnknown) += 1;
    for (Stats* st : {&g_st, &g_tot}) {
        st->plainSets += p.plainSets;
        st->cbRebinds += p.cbRebinds;
        st->cbGame += p.cbGame;
        st->mixed += p.mixed;
        st->lateBinds += p.lateBinds;
        st->lateDraws += p.lateDraws;
        st->mapFailures += p.mapFailures;
        st->viewsFailures += p.viewsFailures;
        st->lateStarts += p.lateStarts;
        st->waitBinds += p.waitBinds;
    }
    TestState& t = g_test;
    ++t.passes;
    t.binds = p.binds;
    t.clears = p.clears;
    t.mapFailures = p.mapFailures;
    t.patchedSets = p.patchedSets;
    t.plainSets = p.plainSets;
    t.cbRebinds = p.cbRebinds;
    t.copiedBack = p.copiedBack;
    t.stashed = (unsigned)s.n;
    t.masks = (unsigned)s.nm;
    t.viewsFailures = p.viewsFailures;
    t.stashValid = s.valid;
    t.jitterKnown = p.jitterKnown;
    t.lateStart = p.lateStarts != 0;
    memcpy(t.rows[0], p.rows[0], sizeof(t.rows[0]));
    memcpy(t.rows[1], p.rows[1], sizeof(t.rows[1]));
    if (g_shot.armed && e == 0 && !g_shot.outerDone) {
        g_shot.pass = p;
        g_shot.pass.slot3 = nullptr;
        g_shot.pass.masks.clear();
        ShotOuter(c);
    }
    ReleasePassRefs(p);
    p = Pass{};
}

// ---------------------------------------------------------------- the focus groups

// Outer UV of the focus window: x_o = (x_f - ox) / sx, u = (x + 1) / 2, v = (1 - y) / 2.
void WindowUv(const Stash& s, float origin[2], float scale[2]) {
    scale[0] = (float)(1 / s.sx);
    scale[1] = (float)(1 / s.sy);
    origin[0] = (float)((1 - 1 / s.sx - s.ox / s.sx) / 2);
    origin[1] = (float)((1 - 1 / s.sy + s.oy / s.sy) / 2);
}

// The texture focus draws a mask into: its clone, or the game's texture when it has none.
ID3D11Texture2D* MaskTarget(ID3D11Resource* orig, ComPtr<ID3D11Texture2D>& hold) {
    if (ID3D11Texture2D* t = FocusClone(orig)) return t;
    if (orig) orig->QueryInterface(IID_PPV_ARGS(&hold));
    return hold.Get();
}

// The focus group's run of its pass can go: the eye's stash of this frame fits its clones.
const char* Decide(int eye) {
    const Stash& s = g_stash[eye];
    const FocusState& f = g_focus[eye];
    if (g_shot.armed) return "shot frame (own pass for the comparison)";
    if (!s.valid || s.frame != g_frame) return "no stash from outer's pass in this frame";
    if (!f.projected || f.frame != g_frame) return "the focus group renders without the focus projection";
    if (!FocusSwapped()) return "the focus clones are not in";
    for (int k = 0; k < s.n; ++k) {
        ID3D11Texture2D* t = FocusClone(s.g[k].orig);
        if (!t) return "a stashed target lost its focus clone";
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        if (d.Width != s.g[k].desc.Width || d.Height != s.g[k].desc.Height || d.Format != s.g[k].desc.Format ||
            d.MipLevels != s.g[k].desc.MipLevels)
            return "a focus clone changed";
    }
    return nullptr;
}

void Fill(ID3D11DeviceContext* c, int eye, bool depth = true) {
    Stash& s = g_stash[eye];
    bool failed = false;
    Internal in;
    for (int k = 0; k < s.n; ++k) {
        if (s.g[k].depth && !depth) continue;
        if (ID3D11Texture2D* t = FocusClone(s.g[k].orig))
            c->CopyResource(t, s.g[k].tex.Get());
        else
            failed = true;
    }
    float origin[2], scale[2];
    WindowUv(s, origin, scale);
    for (int q = 0; q < s.nm; ++q) {
        StashM& m = s.m[q];
        ComPtr<ID3D11Texture2D> hold;
        ID3D11Texture2D* t = MaskTarget(m.orig, hold);
        if (!t) {
            failed = true;
            continue;
        }
        if (m.rtvFor != t || !m.rtv) {
            m.rtv.Reset();
            D3D11_RENDER_TARGET_VIEW_DESC rv{};
            rv.Format = m.fmt;
            rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            if (FAILED(g_g.dev->CreateRenderTargetView(t, &rv, &m.rtv))) {
                failed = true;
                continue;
            }
            m.rtvFor = t;
        }
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        MaskPass(c, m.srv.Get(), m.rtv.Get(), d.Width, d.Height, origin, scale);
        ++g_st.masks;
        ++g_tot.masks;
    }
    ++g_st.fills;
    ++g_tot.fills;
    ++g_test.fills;
    if (failed) {
        ++g_st.fillFailures;
        ++g_tot.fillFailures;
        ++g_test.failures;
    }
}

// ---------------------------------------------------------------- the shot (eye L)

void ShotOuter(ID3D11DeviceContext* c) {
    Shot& sh = g_shot;
    sh.outerDone = true;
    const Stash& s = g_stash[0];
    sh.n = s.valid ? s.n : 0;
    sh.nm = s.valid ? s.nm : 0;
    if (sh.base < 0) return;
    for (int k = 0; k < sh.n; ++k) shots::Grab(c, s.g[k].tex.Get(), sh.base + 2 * k + 1);
}

// Before focus L's own pass in the shot frame: the masks as the module would fill them, at the focus masks' size.
void ShotFocusBefore(ID3D11DeviceContext* c) {
    Shot& sh = g_shot;
    sh.focusBefore = true;
    Stash& s = g_stash[0];
    if (!sh.outerDone || !s.valid || s.frame != g_frame) return;
    float origin[2], scale[2];
    WindowUv(s, origin, scale);
    for (int q = 0; q < sh.nm; ++q) {
        StashM& m = s.m[q];
        ComPtr<ID3D11Texture2D> hold;
        ID3D11Texture2D* t = MaskTarget(m.orig, hold);
        if (!t) continue;
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        d.MiscFlags = 0;
        d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        D3D11_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = m.fmt;
        rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        ComPtr<ID3D11RenderTargetView> rtv;
        if (FAILED(g_g.dev->CreateTexture2D(&d, nullptr, &sh.mask[q])) ||
            FAILED(g_g.dev->CreateRenderTargetView(sh.mask[q].Get(), &rv, &rtv)))
            continue;
        MaskPass(c, m.srv.Get(), rtv.Get(), d.Width, d.Height, origin, scale);
        if (sh.base >= 0) shots::Grab(c, sh.mask[q].Get(), sh.base + 2 * (kMaxTargets + q) + 1);
    }
}

ComPtr<ID3D11Texture2D> StagingOf(ID3D11DeviceContext* c, ID3D11Texture2D* t) {
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    ComPtr<ID3D11Texture2D> s;
    if (FAILED(g_g.dev->CreateTexture2D(&d, nullptr, &s))) return nullptr;
    Internal in;
    c->CopySubresourceRegion(s.Get(), 0, 0, 0, 0, t, 0, nullptr);
    return s;
}

// A shot's R16G16B16A16 target (normal/stencil of the game's TAA; channel 3 high byte = class, 0x88 displays and
// mirrors, 0x86 the player's car): focus's own vs ours per channel, and the classes of the pixels that differ.
void CompareRgba16(ID3D11DeviceContext* c, int k, ID3D11Texture2D* own, ID3D11Texture2D* ours) {
    const ComPtr<ID3D11Texture2D> a = StagingOf(c, own), b = StagingOf(c, ours);
    if (!a || !b) return;
    D3D11_TEXTURE2D_DESC d{};
    a->GetDesc(&d);
    D3D11_MAPPED_SUBRESOURCE ma{}, mb{};
    if (FAILED(c->Map(a.Get(), 0, D3D11_MAP_READ, 0, &ma))) return;
    if (FAILED(c->Map(b.Get(), 0, D3D11_MAP_READ, 0, &mb))) {
        c->Unmap(a.Get(), 0);
        return;
    }
    uint64_t ch[4] = {}, any = 0, cls = 0;
    std::vector<uint32_t> byOwn(256), byOurs(256), total(256);
    for (UINT y = 0; y < d.Height; ++y) {
        const uint16_t* ra = (const uint16_t*)((const uint8_t*)ma.pData + (size_t)y * ma.RowPitch);
        const uint16_t* rb = (const uint16_t*)((const uint8_t*)mb.pData + (size_t)y * mb.RowPitch);
        for (UINT x = 0; x < d.Width; ++x) {
            bool off = false;
            for (int i = 0; i < 4; ++i)
                if (ra[x * 4 + i] != rb[x * 4 + i]) {
                    ++ch[i];
                    off = true;
                }
            const uint32_t ca = ra[x * 4 + 3] >> 8, cb = rb[x * 4 + 3] >> 8;
            ++total[ca];
            any += off;
            if (ca != cb) {
                ++cls;
                ++byOwn[ca];
                ++byOurs[cb];
            }
        }
    }
    c->Unmap(a.Get(), 0);
    c->Unmap(b.Get(), 0);
    auto top = [&](const std::vector<uint32_t>& v) {
        std::vector<int> idx(256);
        for (int i = 0; i < 256; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int x, int y) { return v[x] > v[y]; });
        std::string out;
        for (int i = 0; i < 6 && v[idx[i]]; ++i) {
            char t[48];
            snprintf(t, sizeof(t), "%s0x%02X %u (of %u)", out.empty() ? "" : ", ", idx[i], v[idx[i]], total[idx[i]]);
            out += t;
        }
        return out.empty() ? std::string("none") : out;
    };
    const double px = (double)d.Width * d.Height / 100.0;
    Log("MVR G-buffer shot: target %d (R16G16B16A16) focus L own vs from outer L: pixels off %.3f%%; per channel "
        "%.3f / %.3f / %.3f / %.3f %%; class (channel 3 high byte) off %.3f%%, by focus's own class: %s; by ours: %s", k,
        any / px, ch[0] / px, ch[1] / px, ch[2] / px, ch[3] / px, cls / px, top(byOwn).c_str(), top(byOurs).c_str());
}

// A shot's depth target, bit for bit: focus's own vs ours (float steps; ours farther = larger, the far plane 1).
void CompareDepth(ID3D11DeviceContext* c, int k, ID3D11Texture2D* own, ID3D11Texture2D* ours) {
    const ComPtr<ID3D11Texture2D> a = StagingOf(c, own), b = StagingOf(c, ours);
    if (!a || !b) return;
    D3D11_TEXTURE2D_DESC d{};
    a->GetDesc(&d);
    D3D11_MAPPED_SUBRESOURCE ma{}, mb{};
    if (FAILED(c->Map(a.Get(), 0, D3D11_MAP_READ, 0, &ma))) return;
    if (FAILED(c->Map(b.Get(), 0, D3D11_MAP_READ, 0, &mb))) {
        c->Unmap(a.Get(), 0);
        return;
    }
    uint64_t same = 0, farther = 0, nearer = 0, far1 = 0, bucket[6] = {};  // |steps| 1, 2-4, 5-16, 17-64, 65-1024, more
    int64_t maxFar = 0, maxNear = 0;
    // A systematic shift of ours (px): ours - own ~ g . delta over smooth surfaces (least squares; g = own's central
    // gradient per pixel, pixels whose one-sided differences disagree, silhouettes, are left out).
    double sxx = 0, sxy = 0, syy = 0, sxd = 0, syd = 0, sdd = 0;
    uint64_t fitN = 0;
    auto ownAt = [&](UINT x, UINT y) {
        return *(const float*)((const uint8_t*)ma.pData + (size_t)y * ma.RowPitch + x * 4);
    };
    for (UINT y = 1; y + 1 < d.Height; ++y)
        for (UINT x = 1; x + 1 < d.Width; ++x) {
            const float o = ownAt(x, y), l = ownAt(x - 1, y), r = ownAt(x + 1, y), u = ownAt(x, y - 1),
                        dn = ownAt(x, y + 1);
            if (o >= 1 || l >= 1 || r >= 1 || u >= 1 || dn >= 1) continue;
            const double ax = r - o, bx = o - l, ay = dn - o, by = o - u;
            if (std::fabs(ax - bx) > 0.5 * (std::fabs(ax) + std::fabs(bx)) + 1e-7 ||
                std::fabs(ay - by) > 0.5 * (std::fabs(ay) + std::fabs(by)) + 1e-7)
                continue;
            const double gx = 0.5 * (ax + bx), gy = 0.5 * (ay + by);
            const double dd = (double)*(const float*)((const uint8_t*)mb.pData + (size_t)y * mb.RowPitch + x * 4) - o;
            if (std::fabs(dd) > 4 * (std::fabs(gx) + std::fabs(gy)) + 1e-6) continue;
            sxx += gx * gx;
            sxy += gx * gy;
            syy += gy * gy;
            sxd += gx * dd;
            syd += gy * dd;
            sdd += dd * dd;
            ++fitN;
        }
    const double det = sxx * syy - sxy * sxy;
    const double shx = det > 0 ? (sxd * syy - syd * sxy) / det : 0, shy = det > 0 ? (syd * sxx - sxd * sxy) / det : 0;
    const double resid = fitN ? std::sqrt(std::max(0.0, (sdd - shx * sxd - shy * syd) / fitN)) : 0;
    const double rms = fitN ? std::sqrt(sdd / fitN) : 0;
    for (UINT y = 0; y < d.Height; ++y) {
        const int32_t* ra = (const int32_t*)((const uint8_t*)ma.pData + (size_t)y * ma.RowPitch);
        const int32_t* rb = (const int32_t*)((const uint8_t*)mb.pData + (size_t)y * mb.RowPitch);
        for (UINT x = 0; x < d.Width; ++x) {
            const int64_t steps = (int64_t)rb[x] - ra[x];  // positive floats order as their bits
            if (ra[x] == 0x3F800000) ++far1;
            if (!steps) {
                ++same;
                continue;
            }
            if (steps > 0) {
                ++farther;
                maxFar = std::max(maxFar, steps);
            } else {
                ++nearer;
                maxNear = std::max(maxNear, -steps);
            }
            const int64_t m = steps < 0 ? -steps : steps;
            ++bucket[m <= 1 ? 0 : m <= 4 ? 1 : m <= 16 ? 2 : m <= 64 ? 3 : m <= 1024 ? 4 : 5];
        }
    }
    c->Unmap(a.Get(), 0);
    c->Unmap(b.Get(), 0);
    const double px = (double)d.Width * d.Height / 100.0;
    float slope = 0;
    memcpy(&slope, &g_g.depthCbData[1], sizeof(slope));
    Log("MVR G-buffer shot: target %d (depth, bias %u steps + %.3f px of slope%s) focus L own vs from outer L, bit for bit: same %.3f%%, ours "
        "farther %.3f%% (max %lld steps), nearer %.3f%% (max %lld); |steps| 1: %.3f%%, 2-4: %.3f%%, 5-16: %.3f%%, 17-64: "
        "%.3f%%, 65-1024: %.3f%%, more: %.3f%%; far plane in focus's own %.2f%%",
        k, g_g.depthCbData[0], slope, g_g.depthCbData[4] ? ", 3 x 3 farthest" : "", same / px, farther / px, (long long)maxFar, nearer / px, (long long)maxNear, bucket[0] / px,
        bucket[1] / px, bucket[2] / px, bucket[3] / px, bucket[4] / px, bucket[5] / px, far1 / px);
    Log("MVR G-buffer shot: depth target %d: ours - own fitted as own's slope times a shift over %llu smooth pixels: shift "
        "(%.4f, %.4f) px (ours = own sampled there), rms %.3g, left after the shift %.3g; the pass's jitter offset "
        "(%.4f, %.4f), M %s",
        k, (unsigned long long)fitN, shx, shy, rms, resid, g_shot.pass.jit[0], g_shot.pass.jit[1],
        g_shot.pass.jitterKnown ? "moved to focus's own" : "with outer's jitter");
}

// After focus L's own pass: its targets.
void ShotFocusAfter(ID3D11DeviceContext* c) {
    Shot& sh = g_shot;
    sh.focusDone = true;
    const Stash& s = g_stash[0];
    for (int k = 0; k < sh.n; ++k) {
        ID3D11Texture2D* t = FocusClone(s.g[k].orig);
        const DXGI_FORMAT f = s.g[k].desc.Format;
        if (t && s.g[k].tex && f >= DXGI_FORMAT_R16G16B16A16_TYPELESS && f <= DXGI_FORMAT_R16G16B16A16_SINT)
            CompareRgba16(c, k, t, s.g[k].tex.Get());
        if (t && s.g[k].tex && s.g[k].depth && (f == DXGI_FORMAT_R32_TYPELESS || f == DXGI_FORMAT_D32_FLOAT))
            CompareDepth(c, k, t, s.g[k].tex.Get());
        const D3D11_TEXTURE2D_DESC& d = s.g[k].desc;
        snprintf(g_shotLabels[k], sizeof(g_shotLabels[k]), "MVR G-buffer %d (%ux%u fmt %d%s): focus L vs from outer L", k,
                 d.Width, d.Height, (int)d.Format, s.g[k].depth ? ", depth" : "");
        if (t && sh.base >= 0) shots::Grab(c, t, sh.base + 2 * k);
    }
    for (int q = 0; q < sh.nm; ++q) {
        ComPtr<ID3D11Texture2D> hold;
        ID3D11Texture2D* t = MaskTarget(s.m[q].orig, hold);
        D3D11_TEXTURE2D_DESC d{};
        if (t) t->GetDesc(&d);
        snprintf(g_shotLabels[kMaxTargets + q], sizeof(g_shotLabels[0]),
                 "MVR mask %d (%ux%u fmt %d): focus L vs outer L's in the window", q, d.Width, d.Height, (int)s.m[q].fmt);
        if (t && sh.base >= 0) shots::Grab(c, t, sh.base + 2 * (kMaxTargets + q));
    }
    ReportShot();
}


void ReportShot() {
    Shot& sh = g_shot;
    if (!sh.armed || sh.reported) return;
    sh.reported = true;
    const Pass& p = sh.pass;
    if (!sh.outerDone) {
        Log("MVR G-buffer shot (frame %llu): no comparison (outer L's pass did not run with MVR in the shot frame)",
            (unsigned long long)sh.frame);
        return;
    }
    std::string pl;
    for (const auto& [sum, n] : sh.plain) {
        char b[80];
        snprintf(b, sizeof(b), "%s%s x%u", pl.empty() ? "" : ", ", sum.empty() ? "(not seen by the patch)" : sum.c_str(),
                 n);
        pl += b;
    }
    const std::string state = p.failed ? "FAILED: " + p.why : "ok";
    Log("MVR G-buffer shot (frame %llu): outer L's pass %s; %d outer-size target(s) stashed, %d mask(s); binds %u "
        "(tail %u), clears %u, map failures %u, mixed binds %u, late binds %u; VS sets patched %u, unpatched %u%s%s; "
        "draws %u (with one view %u); slot 3 rebinds %u (game buffers %u); copied back %u; SetMultiViewMode failures %u",
        (unsigned long long)sh.frame, state.c_str(), sh.n, sh.nm, p.binds, p.tailBinds, p.clears, p.mapFailures,
        p.mixed, p.lateBinds, p.patchedSets, p.plainSets, pl.empty() ? "" : ": ", pl.c_str(), p.draws, p.plainDraws,
        p.cbRebinds, p.cbGame, p.copiedBack, p.viewsFailures);
    const char* jit = p.jitterKnown ? "moved to focus's own" : Cfg().mvrJitterFix ? "UNKNOWN (outer's kept)"
                                                                                  : "outer's (mvr_jitter_fix=0)";
    Log("MVR G-buffer shot: view 1 = focus L in the corner %u x %u of %u x %u; jitter %s (offset %.4f, %.4f); M rows "
        "[%.6f 0 0 %.6f] [0 %.6f 0 %.6f]; focus L ran its own pass for the comparison: %s",
        p.fw, p.fh, p.w, p.h, jit, p.jit[0], p.jit[1], p.rows[0][0], p.rows[0][3], p.rows[1][1], p.rows[1][3],
        sh.focusDone ? "yes" : "NO");
}

// ---------------------------------------------------------------- D3D11 hooks

using VSSetShaderFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11VertexShader*, ID3D11ClassInstance* const*,
                                               UINT);
using VSSetCbFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11Buffer* const*);
using VSSetCb1Fn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext1*, UINT, UINT, ID3D11Buffer* const*, const UINT*,
                                            const UINT*);
using DrawIndexedInstancedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using DrawInstancedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
using DrawAutoFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*);
using DrawIndirectFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Buffer*, UINT);
VSSetShaderFn o_vsSet = nullptr;
VSSetCbFn o_vsCb = nullptr;
VSSetCb1Fn o_vsCb1 = nullptr;
DrawIndexedInstancedFn o_dii = nullptr;
DrawInstancedFn o_di = nullptr;
DrawAutoFn o_da = nullptr;
DrawIndirectFn o_diii = nullptr, o_dii2 = nullptr;
bool g_d3dHooked = false;

bool InPass(ID3D11DeviceContext* c) { return t_pass && !t_internal && c == Imm(); }

void OnVs(ID3D11DeviceContext* c, ID3D11VertexShader* vs) {
    Protect("VSSetShader", [&] {
        Pass& p = g_p;
        if (!vs) return;
        const bool shot = g_shot.armed && p.eye == 0;
        char sum[40] = "";
        const bool patched = mvrvs::IsPatched(vs, shot ? sum : nullptr);
        NvU32 want = 1;
        if (patched) {
            ++p.patchedSets;
            FillM(c);
            want = 2;
        } else {
            ++p.plainSets;
            if (shot) {
                auto it = std::find_if(g_shot.plain.begin(), g_shot.plain.end(),
                                       [&](const auto& e) { return e.first == sum; });
                if (it != g_shot.plain.end())
                    ++it->second;
                else if (g_shot.plain.size() < 8)
                    g_shot.plain.push_back({std::string(sum), 1u});
            }
        }
        if (want != p.views && SetViews(c, want)) p.views = want;
    });
}

// The game set VS slot 3 (null when it clears a range): noted as its binding, the module's buffer goes back.
void OnSlot3(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT first, UINT num) {
    Protect("VSSetConstantBuffers", [&] {
        Pass& p = g_p;
        ID3D11Buffer* cb = g_g.rowsCb.Get();
        if (b == cb) return;
        if (b) {
            ++p.cbGame;
            b->AddRef();
        }
        if (p.slot3) p.slot3->Release();
        p.slot3 = b;
        p.slot3First = first;
        p.slot3Num = num;
        ++p.cbRebinds;
        o_vsCb(c, mvrvs::kCbSlot, 1, &cb);
    });
}

void STDMETHODCALLTYPE D_VSSetShader(ID3D11DeviceContext* c, ID3D11VertexShader* vs, ID3D11ClassInstance* const* ci,
                                     UINT n) {
    o_vsSet(c, vs, ci, n);
    if (InPass(c)) OnVs(c, vs);
}
void STDMETHODCALLTYPE D_VSSetCb(ID3D11DeviceContext* c, UINT start, UINT num, ID3D11Buffer* const* b) {
    o_vsCb(c, start, num, b);
    if (InPass(c) && start <= mvrvs::kCbSlot && start + num > mvrvs::kCbSlot)
        OnSlot3(c, b ? b[mvrvs::kCbSlot - start] : nullptr, 0, 0);
}
void STDMETHODCALLTYPE D_VSSetCb1(ID3D11DeviceContext1* c, UINT start, UINT num, ID3D11Buffer* const* b,
                                  const UINT* first, const UINT* cnt) {
    o_vsCb1(c, start, num, b, first, cnt);
    if (InPass(c) && start <= mvrvs::kCbSlot && start + num > mvrvs::kCbSlot) {
        const UINT i = mvrvs::kCbSlot - start;
        OnSlot3(c, b ? b[i] : nullptr, first ? first[i] : 0, cnt ? cnt[i] : 0);
    }
}

void STDMETHODCALLTYPE D_DrawIndexedInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, INT e, UINT f) {
    BeforeDraw(c);
    o_dii(c, a, b, d, e, f);
}
void STDMETHODCALLTYPE D_DrawInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, UINT e) {
    BeforeDraw(c);
    o_di(c, a, b, d, e);
}
void STDMETHODCALLTYPE D_DrawAuto(ID3D11DeviceContext* c) {
    BeforeDraw(c);
    o_da(c);
}
void STDMETHODCALLTYPE D_DrawIndexedInstancedIndirect(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT o) {
    BeforeDraw(c);
    o_diii(c, b, o);
}
void STDMETHODCALLTYPE D_DrawInstancedIndirect(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT o) {
    BeforeDraw(c);
    o_dii2(c, b, o);
}

bool CheckVtable(uintptr_t base, uintptr_t vt, uintptr_t exec, const char* name) {
    uintptr_t e = 0, f = 0;
    if (hooks::Readable((void*)(base + vt), kVtSlots * 8) &&
        hooks::TryRead((void*)(base + vt + kExecuteSlot * 8), &e, 8) &&
        hooks::TryRead((void*)(base + vt + kFinishSlot * 8), &f, 8) && e == base + exec && f == base + kTaskFinish)
        return true;
    Log("MVR G-buffer: %s vtable does not match (Execute at AMS2AVX.exe+0x%llX, Finish +0x%llX; expected +0x%llX, "
        "+0x%llX); off",
        name, (unsigned long long)(e ? e - base : 0), (unsigned long long)(f ? f - base : 0), (unsigned long long)exec,
        (unsigned long long)kTaskFinish);
    return false;
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    Protect("Install", [&] {
        const uintptr_t base = hooks::ExeBase();
        if (!base || !CheckVtable(base, kPhase1Vtable, kPhase1Execute, "CPhase1Render") ||
            !CheckVtable(base, kBuildVtable, kBuildExecute, "CPhase1BuildCommandThread"))
            return;
        g_phase1Vt = base + kPhase1Vtable;
        g_buildVt = base + kBuildVtable;
        g_installed = true;
        Log("MVR G-buffer: on; outer L's and outer R's G-buffer pass draw the eye's focus too (MVR, view 1 in the "
            "corner of slice 1 of two-slice copies), the focus groups skip CPhase1BuildCommandThread and CPhase1Render "
            "and get it in their clones; jitter fix %s; hotkey 0x%X+0x%X",
            Cfg().mvrJitterFix ? "on" : "off", Cfg().mvrKeyMod, Cfg().mvrKey);
    });
    g_on = g_installed;
    return g_installed;
}

void InstallD3DHooks(ID3D11DeviceContext* imm) {
    if (!g_installed || g_d3dHooked || !imm) return;
    g_d3dHooked = true;
    Protect("InstallD3DHooks", [&] {
        void** vt = *(void***)imm;
        // ID3D11DeviceContext vtable slots (d3d11.h order); VSSetConstantBuffers1 is ID3D11DeviceContext1's 119.
        const bool a = hooks::CreateAt(vt[11], (void*)D_VSSetShader, (void**)&o_vsSet, "VSSetShader");
        const bool b = hooks::CreateAt(vt[7], (void*)D_VSSetCb, (void**)&o_vsCb, "VSSetConstantBuffers");
        bool b1 = false;
        ComPtr<ID3D11DeviceContext1> c1;
        if (SUCCEEDED(imm->QueryInterface(IID_PPV_ARGS(&c1))))
            b1 = hooks::CreateAt((*(void***)c1.Get())[119], (void*)D_VSSetCb1, (void**)&o_vsCb1,
                                 "VSSetConstantBuffers1");
        const bool d = hooks::CreateAt(vt[20], (void*)D_DrawIndexedInstanced, (void**)&o_dii, "DrawIndexedInstanced");
        const bool e = hooks::CreateAt(vt[21], (void*)D_DrawInstanced, (void**)&o_di, "DrawInstanced");
        const bool f = hooks::CreateAt(vt[38], (void*)D_DrawAuto, (void**)&o_da, "DrawAuto");
        const bool g = hooks::CreateAt(vt[39], (void*)D_DrawIndexedInstancedIndirect, (void**)&o_diii,
                                       "DrawIndexedInstancedIndirect");
        const bool h = hooks::CreateAt(vt[40], (void*)D_DrawInstancedIndirect, (void**)&o_dii2,
                                       "DrawInstancedIndirect");
        Log("MVR G-buffer: D3D11 hooks: VSSetShader %s, VSSetConstantBuffers %s, VSSetConstantBuffers1 %s, draws (shot "
            "counts) %s",
            a ? "ok" : "FAILED", b ? "ok" : "FAILED", b1 ? "ok" : c1 ? "FAILED" : "(no ID3D11DeviceContext1)",
            d && e && f && g && h ? "ok" : "FAILED");
        if (!a || !b || (c1 && !b1)) {
            g_on = false;
            Log("MVR G-buffer: off (a VS hook is missing: MVR could reach a VS without view 1, or slot 3 a game "
                "buffer)");
        }
    });
}

void SetJitterSource(JitterFn fn) { g_jitterFn = fn; }

bool On() { return g_on.load(std::memory_order_relaxed); }

void OnFrameStart(bool enabled) {
    if (!g_installed) return;
    Protect("OnFrameStart", [&] {
        const uint32_t gen = gpu::Generation();
        if (gen != g_gen && !g_testCtx) {
            g_gen = gen;
            ReleaseAll();
        }
        ++g_frame;
        if (g_p.part != PartNone) {  // a pass whose task did not end
            t_pass = false;
            ReleasePassRefs(g_p);
            g_p = Pass{};
        }
        if (g_shot.armed) {
            ReportShot();
            g_shot = Shot{};
        }
        const bool on = enabled && On();
        if (on != g_enabled)
            Log("MVR G-buffer: %s (frame %llu)", on ? "ON" : "OFF (the focus groups run their own pass)",
                (unsigned long long)g_frame);
        g_enabled = on;
        for (FocusState& f : g_focus) f.decided = f.skip = false;
        ++g_st.frames;
        ++g_tot.frames;
    });
}

int PrepareShot(const char** names, int base, int max, shots::Pair* pairs, int maxPairs, int* pairCount) {
    *pairCount = 0;
    if (!On()) return 0;
    int n = 0;
    Protect("PrepareShot", [&] {
        if (!g_enabled) {
            Log("MVR G-buffer: no comparison in this shot (off by the hotkey)");
            return;
        }
        if (max < kShotImages || maxPairs < kMaxTargets + kMaxMasks) {
            Log("MVR G-buffer: no comparison in this shot (no room for its images)");
            return;
        }
        Shot& sh = g_shot;
        sh = Shot{};
        sh.armed = true;
        sh.base = base;
        sh.frame = g_frame;
        for (int k = 0; k < kMaxTargets + kMaxMasks; ++k) {
            const bool mask = k >= kMaxTargets;
            const int j = mask ? k - kMaxTargets : k;
            snprintf(g_shotNames[2 * k], sizeof(g_shotNames[0]), "mvr_%s%d_focus", mask ? "m" : "g", j);
            snprintf(g_shotNames[2 * k + 1], sizeof(g_shotNames[0]), "mvr_%s%d_ours", mask ? "m" : "g", j);
            names[base + 2 * k] = g_shotNames[2 * k];
            names[base + 2 * k + 1] = g_shotNames[2 * k + 1];
            snprintf(g_shotLabels[k], sizeof(g_shotLabels[k]), "MVR %s %d: focus L vs %s", mask ? "mask" : "G-buffer", j,
                     mask ? "outer L's in the window" : "from outer L");
            pairs[k] = {base + 2 * k, base + 2 * k + 1, g_shotLabels[k]};
        }
        *pairCount = kMaxTargets + kMaxMasks;
        n = kShotImages;
    });
    return n;
}

void CancelShot() {
    if (g_shot.armed) Protect("CancelShot", [&] { g_shot = Shot{}; });
}

void SetWindow(int eye, bool valid, double sx, double ox, double sy, double oy, uint32_t width, uint32_t height) {
    if (eye < 0 || eye > 1) return;
    Win& w = g_win[eye];
    w.valid = valid;
    w.frame = g_frame;
    w.sx = sx;
    w.ox = ox;
    w.sy = sy;
    w.oy = oy;
    w.w = width;
    w.h = height;
}

void SetMode(Mode m) { g_mode = m; }
Mode GetMode() { return g_mode; }
const char* ModeName(Mode m) {
    switch (m) {
    case Mode::Skip:
        return "SKIP (normal B8: focus skips its pass and gets outer's, the depth made conservative by mvr_depth_dilate, "
               "mvr_depth_slope, mvr_depth_bias)";
    case Mode::SkipRaw: return "SKIP RAW (focus gets outer's with the depth as MVR drew it: flickers)";
    case Mode::NoDepth: return "NO DEPTH (focus runs its own pass, then outer's on top except the depth)";
    case Mode::OuterOnly: return "OUTER ONLY (outer's pass with MVR, focus runs its own and gets nothing)";
    case Mode::Overwrite: return "OVERWRITE (focus runs its own pass, then all of outer's on top)";
    }
    return "?";
}

void FocusGroup(int eye, bool projected) {
    if (eye < 0 || eye > 1) return;
    g_focus[eye].projected = projected;
    g_focus[eye].frame = g_frame;
}

bool BeforeTask(uint8_t* task, int group) {
    if (!On() || !task || group < kOuterL || group > kFocusR) return false;
    uintptr_t vt = 0;
    if (!hooks::TryRead(task, &vt, 8) || (vt != g_phase1Vt && vt != g_buildVt)) return false;
    bool skip = false;
    Protect("BeforeTask", [&] {
        if (group <= kOuterR) {
            if (vt == g_phase1Vt) PassBegin(task, group);
            return;
        }
        const int eye = group - kFocusL;
        FocusState& f = g_focus[eye];
        if (!f.decided) {
            f.decided = true;
            const char* why = g_enabled ? Decide(eye) : "off";
            f.skip = why == nullptr;
            if (why && g_enabled) Own(eye, why);
            if (g_lastSkip[eye] >= 0 && g_lastSkip[eye] != (int)f.skip) {
                ++g_st.switches[eye];
                ++g_tot.switches[eye];
            }
            g_lastSkip[eye] = f.skip;
        }
        if (f.skip && g_mode != Mode::Skip && g_mode != Mode::SkipRaw) {  // the engine runs both tasks
            const bool fill = g_mode == Mode::Overwrite || g_mode == Mode::NoDepth;
            if (vt == g_phase1Vt && fill) {  // the stash goes on top after CPhase1Render
                t_fillTask = task;
                t_fillEye = eye;
                t_fillDepth = g_mode != Mode::NoDepth;
            }
            return;
        }
        if (f.skip) {
            if (vt == g_phase1Vt) {
                Fill(Imm(), eye);
                ++g_st.skipped[eye];
                ++g_tot.skipped[eye];
            }
            ++g_test.focusSkipped;
            skip = true;
            return;
        }
        if (vt == g_phase1Vt && eye == 0 && g_shot.armed && !g_shot.focusBefore) {
            ShotFocusBefore(Imm());
            t_focusTask = task;
        }
    });
    return skip;
}

void AfterTask(uint8_t* task) {
    if (!task) return;
    if (task == t_fillTask) {
        const int eye = t_fillEye;
        t_fillTask = nullptr;
        t_fillEye = -1;
        Protect("AfterTask", [&] {
            Fill(Imm(), eye, t_fillDepth);
            ++g_st.overwrites[eye];
            ++g_tot.overwrites[eye];
        });
        return;
    }
    if (task == t_focusTask) {
        t_focusTask = nullptr;
        Protect("AfterTask", [&] { ShotFocusAfter(Imm()); });
        return;
    }
    if (g_p.part == PartNone || task != g_p.task) return;
    Protect("AfterTask", [&] { PassEnd(Imm()); });
}

bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut) {
    if (t_internal || g_p.part == PartNone || c != Imm()) return false;
    bool ok = false;
    Protect("MapTargets", [&] {
        Pass& p = g_p;
        if (p.part == PartTail) {
            NoteTail(n, rtv, dsv);
            return;
        }
        if (p.part == PartWait) {
            const ComPtr<ID3D11Resource> main = MainOf(n, rtv, dsv);
            if (!main || !StartAtFirst(c, main.Get())) return;  // an unbind waits on; a refused start runs plain
        }
        if (!t_pass) return;
        ++p.binds;
        bool any = dsv != nullptr, outer = dsv && OuterSize(dsv);
        unsigned mixed = 0;
        for (UINT i = 0; i < n && rtv; ++i)
            if (rtv[i]) {
                any = true;
                if (OuterSize(rtv[i]))
                    outer = true;
                else
                    ++mixed;
            }
        if (any && !outer) {  // the masks: the G-buffer is done
            EndMvr(c, false);
            NoteTail(n, rtv, dsv);
            return;
        }
        p.mixed += mixed;
        for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
            rtvOut[i] = rtv && rtv[i] ? ArrayRtv(c, rtv[i]) : nullptr;
        *dsvOut = dsv ? ArrayDsv(c, dsv) : nullptr;
        bool failed = dsv && !*dsvOut;
        for (UINT i = 0; i < n && rtv; ++i) failed |= rtv[i] && !rtvOut[i];
        if (failed) {  // a view without its copy: no MVR for the rest (view 1 would land in slice 0)
            EndMvr(c, false);
            return;
        }
        ok = true;
    });
    return ok;
}

// A clear before the first bind of a waiting pass starts the MVR part (its copies get the clear).
bool StartAtClear(ID3D11DeviceContext* c, ID3D11View* v) {
    if (g_p.part != PartWait || t_internal || !v || c != Imm()) return false;
    bool started = false;
    Protect("StartAtClear", [&] {
        ComPtr<ID3D11Resource> main;
        v->GetResource(&main);
        started = StartAtFirst(c, main.Get());
    });
    return started;
}

ID3D11RenderTargetView* MapClear(ID3D11DeviceContext* c, ID3D11RenderTargetView* v) {
    StartAtClear(c, v);
    if (!InPass(c) || !v) return v;
    ID3D11RenderTargetView* out = v;
    Protect("ClearRenderTargetView", [&] {
        ++g_p.clears;
        if (ID3D11RenderTargetView* m = ArrayRtv(c, v)) out = m;
    });
    return out;
}

ID3D11DepthStencilView* MapClear(ID3D11DeviceContext* c, ID3D11DepthStencilView* v) {
    StartAtClear(c, v);
    if (!InPass(c) || !v) return v;
    ID3D11DepthStencilView* out = v;
    Protect("ClearDepthStencilView", [&] {
        ++g_p.clears;
        if (ID3D11DepthStencilView* m = ArrayDsv(c, v)) out = m;
    });
    return out;
}

bool BeforeDraw(ID3D11DeviceContext* c) {
    if (g_p.part == PartTail && !t_internal && c == Imm()) {
        ++g_p.tailDraws;
        if (g_p.lateBound) ++g_p.lateDraws;
    }
    if (!t_pass || t_internal || !g_shot.armed || c != Imm() || g_p.eye != 0) return false;
    ++g_p.draws;
    if (g_p.views != 2) ++g_p.plainDraws;
    return false;
}

void AfterDraw(ID3D11DeviceContext*) {}

std::string Describe() {
    if (!g_installed) return "MVR G-buffer off";
    char buf[160];
    snprintf(buf, sizeof(buf), "MVR G-buffer %s (focus pass from outer: L %llu, R %llu of %llu frames)",
             g_dead ? "STOPPED" : !On() ? "off (hooks)" : g_enabled ? "on" : "off (hotkey)",
             (unsigned long long)g_tot.skipped[0], (unsigned long long)g_tot.skipped[1],
             (unsigned long long)g_tot.frames);
    return buf;
}

void LogStats(bool totals) {
    if (!g_installed) return;
    Stats& s = totals ? g_tot : g_st;
    if (!s.frames) return;
    std::string rs;
    for (const Reason& r : s.reasons) {
        char b[200];
        snprintf(b, sizeof(b), "%s%s %llu", rs.empty() ? "" : "; ", r.text.c_str(), (unsigned long long)r.n);
        rs += b;
    }
    auto ms = [&](int e) { return s.passes[e] ? s.passMs[e] / s.passes[e] : 0.0; };
    Log("MVR G-buffer%s (%llu frames): outer passes L %llu / R %llu (no stash %llu / %llu), CPU per pass L %.3f / R "
        "%.3f ms, started at the first bind %llu (after %llu other binds); focus choice changes L %llu / R %llu; own pass with outer's on top (diagnosis modes) L %llu / R %llu; focus G-buffer from outer L %llu / R %llu, own pass L %llu / R %llu%s%s%s; fills %llu (failed %llu), "
        "masks %llu; jitter moved in %llu passes (unknown %llu); unpatched VS sets %llu, slot 3 rebinds %llu (game "
        "buffers %llu), mixed binds %llu, late binds %llu (draws after them %llu), map failures %llu, SetMultiViewMode failures %llu",
        totals ? " (session)" : "", (unsigned long long)s.frames, (unsigned long long)s.passes[0],
        (unsigned long long)s.passes[1], (unsigned long long)s.passFailed[0], (unsigned long long)s.passFailed[1],
        ms(0), ms(1), (unsigned long long)s.lateStarts, (unsigned long long)s.waitBinds,
        (unsigned long long)s.switches[0], (unsigned long long)s.switches[1],
        (unsigned long long)s.overwrites[0], (unsigned long long)s.overwrites[1],
        (unsigned long long)s.skipped[0], (unsigned long long)s.skipped[1],
        (unsigned long long)s.own[0], (unsigned long long)s.own[1], rs.empty() ? "" : " (", rs.c_str(),
        rs.empty() ? "" : ")", (unsigned long long)s.fills, (unsigned long long)s.fillFailures,
        (unsigned long long)s.masks, (unsigned long long)s.jitterKnown, (unsigned long long)s.jitterUnknown,
        (unsigned long long)s.plainSets, (unsigned long long)s.cbRebinds, (unsigned long long)s.cbGame,
        (unsigned long long)s.mixed, (unsigned long long)s.lateBinds, (unsigned long long)s.lateDraws,
        (unsigned long long)s.mapFailures,
        (unsigned long long)s.viewsFailures);
    if (!totals) s = Stats{};
}

void Shutdown() {
    if (!g_installed) return;
    Protect("Shutdown", [&] { ReleaseAll(); });
}

void TestSetup(ID3D11DeviceContext* ctx, const void* phase1Vtable, const void* buildVtable, CloneFn clone,
               JitterFn jitter) {
    g_testCtx = ctx;
    g_phase1Vt = (uintptr_t)phase1Vtable;
    g_buildVt = (uintptr_t)buildVtable;
    g_cloneFn = clone;
    g_jitterFn = jitter;
    g_installed = true;
    g_on = true;
    InstallD3DHooks(ctx);
}

TestState TestGet() { return g_test; }

void TestDepthBias(int steps, float slope, bool dilate) {
    g_testBias = steps;
    g_testSlope = slope;
    g_testDilate = dilate;
}
ID3D11Texture2D* TestArray(int k) { return k >= 0 && k < (int)g_copies.size() ? g_copies[k].arr.Get() : nullptr; }

void TestEnd() {
    const bool faulted = g_test.faulted;
    g_test = TestState{};
    g_test.faulted = faulted;
}

}  // namespace qvfr::mvrpass
