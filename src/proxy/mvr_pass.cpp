#include "mvr_pass.h"

#include <d3d11_1.h>
#include <wrl/client.h>

#include <nvapi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "clones.h"
#include "common.h"
#include "gpu.h"
#include "hooks.h"
#include "mvr_pass_blit_ps.h"      // kMvrPassBlitPS
#include "mvr_pass_color_in_ms_ps.h"   // kMvrPassColorInMsPS (B12)
#include "mvr_pass_color_out_ms_ps.h"  // kMvrPassColorOutMsPS
#include "mvr_pass_depth_in_ms_ps.h"   // kMvrPassDepthInMsPS
#include "mvr_pass_depth_ms_ps.h"      // kMvrPassDepthMsPS
#include "mvr_pass_depth_in_ps.h"  // kMvrPassDepthInPS
#include "mvr_pass_depth_ps.h"     // kMvrPassDepthPS (fxc from shaders/mvr_pass.hlsl)
#include "mvr_pass_mask_ps.h"   // kMvrPassMaskPS
#include "mvr_pass_vs.h"        // kMvrPassVS
#include "mvr_ps.h"
#include "mvr_ps_patch.h"  // dxbc::kMvrPsMagic
#include "mvr_vs.h"

namespace qvfr::mvrpass {

namespace {

using Microsoft::WRL::ComPtr;

// Engine layout (docs/mvr.md B5, B8), AMS2AVX.exe build aa43bd68.
constexpr uintptr_t kPhase1Vtable = 0x204adc8;  // CPhase1Render (RVA)
constexpr uintptr_t kPhase1Execute = 0x9c96c0;
constexpr uintptr_t kBuildVtable = 0x204ae20;   // CPhase1BuildCommandThread
constexpr uintptr_t kBuildExecute = 0x9c9310;
constexpr uintptr_t kTaskFinish = 0xcde260;     // +0x38 of both (and of the tasks below)
// MVR forward step 1 (B9a, Ghidra): the eye group's SSAO, lights and forward tasks.
constexpr uintptr_t kSsaoVtable = 0x204c618;    // CSSAORender
constexpr uintptr_t kSsaoExecute = 0x9d6580;
constexpr uintptr_t kPhase2Vtable = 0x204bef8;  // CPhase2Render (deferred lights)
constexpr uintptr_t kPhase2Execute = 0x9ce9c0;
constexpr uintptr_t kPhase3Vtable = 0x204af30;  // CPhase3Render (opaque forward)
constexpr uintptr_t kPhase3Execute = 0x9cac80;
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
uintptr_t g_ssaoVt = 0, g_phase2Vt = 0, g_phase3Vt = 0;  // 0: their vtables did not match (no early step)
int g_early = 0;  // [debug] mvr_early and its hotkey (mvr_pass.h)
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
// B12 step 5: the slice of the engine target's array that is focus's clone for `eye` (0: a clone of its own).
SliceFn g_sliceFn = nullptr;  // the smoke's stand-in for clones::FocusSlice
int FocusSliceOf(ID3D11Resource* orig, int eye, uint32_t* w = nullptr, uint32_t* h = nullptr) {
    if (!orig) return 0;
    if (g_cloneFn) return g_sliceFn ? g_sliceFn(orig, eye, w, h) : 0;
    return clones::FocusSlice((ID3D11Resource*)clones::Original(orig), eye, w, h);
}
bool FrameJitter(float j[2]) { return g_jitterFn && g_jitterFn(j); }

thread_local bool t_pass = false;      // the MVR part of a pass runs on this thread
thread_local bool t_internal = false;  // the module's own D3D11 calls
thread_local uint8_t* t_focusTask = nullptr;  // focus L's CPhase1Render running in a shot frame
thread_local uint8_t* t_fillTask = nullptr;   // overwrite mode: a focus CPhase1Render whose clones get the stash after it
thread_local int t_fillEye = -1;
thread_local bool t_fillDepth = true;
thread_local uint8_t* t_fwdFocusTask = nullptr;  // focus L's CPhase3Render in a forward test shot frame (B9c)

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
    ComPtr<ID3D11PixelShader> psDepth, psMask, psDepthIn, psBlit;
    ComPtr<ID3D11PixelShader> psColorInMs, psColorOutMs, psDepthInMs, psDepthMs;  // B12: per sample
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11DepthStencilState> dssWrite, dssOff;
    ComPtr<ID3D11RasterizerState> rs;
    ComPtr<ID3D11Buffer> rowsCb, maskCb, depthCb;
    ComPtr<ID3D11Buffer> psRowsCb;  // B9c: the patched pixel shaders' rows (S and O, UV scale, magic) in PS slot 3
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
    if (SUCCEEDED(hr))
        hr = g.dev->CreatePixelShader(kMvrPassDepthInPS, sizeof(kMvrPassDepthInPS), nullptr, &g.psDepthIn);
    if (SUCCEEDED(hr)) hr = g.dev->CreatePixelShader(kMvrPassBlitPS, sizeof(kMvrPassBlitPS), nullptr, &g.psBlit);
    if (SUCCEEDED(hr))
        hr = g.dev->CreatePixelShader(kMvrPassColorInMsPS, sizeof(kMvrPassColorInMsPS), nullptr, &g.psColorInMs);
    if (SUCCEEDED(hr))
        hr = g.dev->CreatePixelShader(kMvrPassColorOutMsPS, sizeof(kMvrPassColorOutMsPS), nullptr, &g.psColorOutMs);
    if (SUCCEEDED(hr))
        hr = g.dev->CreatePixelShader(kMvrPassDepthInMsPS, sizeof(kMvrPassDepthInMsPS), nullptr, &g.psDepthInMs);
    if (SUCCEEDED(hr)) hr = g.dev->CreatePixelShader(kMvrPassDepthMsPS, sizeof(kMvrPassDepthMsPS), nullptr, &g.psDepthMs);
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
    bd.ByteWidth = 4 * 4 * sizeof(float);
    if (SUCCEEDED(hr)) hr = g.dev->CreateBuffer(&bd, nullptr, &g.psRowsCb);
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

// A PS constant buffer slot with its D3D11.1 range: the game binds through PSSetConstantBuffers1 with offsets into one
// large buffer (the sky dome's _GLOBAL_ in slot 0, B12), so a restore without the range makes its next draw read the
// buffer's start (the dark sky of B12 build 1).
struct PsCb {
    ID3D11Buffer* b = nullptr;
    UINT first = 0, num = 0;
    void Save(ID3D11DeviceContext* c, UINT slot) {
        if (g_g.c1)
            g_g.c1->PSGetConstantBuffers1(slot, 1, &b, &first, &num);
        else
            c->PSGetConstantBuffers(slot, 1, &b);
    }
    // Sets it back and releases the reference.
    void Restore(ID3D11DeviceContext* c, UINT slot) {
        if (g_g.c1 && b && (first || num))
            g_g.c1->PSSetConstantBuffers1(slot, 1, &b, &first, &num);
        else
            c->PSSetConstantBuffers(slot, 1, &b);
        if (b) b->Release();
        b = nullptr;
    }
};

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
    PsCb cb;
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
        cb.Save(c, 0);
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
        cb.Restore(c, 0);
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

// The depth of slice 1's corner (w x h) into `dsv` (`ps`: PSDepth by default, PSDepthMs for a multisampled depth).
void DepthPass(ID3D11DeviceContext* c, ID3D11ShaderResourceView* src, ID3D11DepthStencilView* dsv, UINT w, UINT h,
               uint32_t biasUlps, float slope, bool dilate, ID3D11PixelShader* ps = nullptr) {
    Internal in;
    uint32_t v[8] = {biasUlps, 0, w, h, dilate ? 1u : 0u, 0, 0, 0};
    memcpy(&v[1], &slope, sizeof(slope));
    if (memcmp(v, g_g.depthCbData, sizeof(v)) != 0) {
        c->UpdateSubresource(g_g.depthCb.Get(), 0, nullptr, v, 0, 0);
        memcpy(g_g.depthCbData, v, sizeof(v));
    }
    Saved s;
    s.Save(c);
    PsCb cb1;
    cb1.Save(c, 1);
    c->OMSetRenderTargets(0, nullptr, dsv);
    c->PSSetShaderResources(0, 1, &src);
    ID3D11Buffer* cb = g_g.depthCb.Get();
    c->PSSetConstantBuffers(1, 1, &cb);
    FullScreen(c, ps ? ps : g_g.psDepth.Get(), g_g.dssWrite.Get(), w, h);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    cb1.Restore(c, 1);
    s.Restore(c);
}

// B12: `src` sample by sample into `rtv` over w x h (`ps`: PSColorInMs or PSColorOutMs).
void ColorPassMs(ID3D11DeviceContext* c, ID3D11ShaderResourceView* src, ID3D11RenderTargetView* rtv, UINT w, UINT h,
                 ID3D11PixelShader* ps) {
    Internal in;
    Saved s;
    s.Save(c);
    c->OMSetRenderTargets(1, &rtv, nullptr);
    c->PSSetShaderResources(0, 1, &src);
    FullScreen(c, ps, g_g.dssOff.Get(), w, h);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    s.Restore(c);
}

// A mask of outer read in the focus window into `rtv` (w x h): outer UV = origin + focus UV * scale. `ps`: PSMask by
// default, PSBlit for a color texture.
void MaskPass(ID3D11DeviceContext* c, ID3D11ShaderResourceView* src, ID3D11RenderTargetView* rtv, UINT w, UINT h,
              const float origin[2], const float scale[2], ID3D11PixelShader* ps = nullptr) {
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
    FullScreen(c, ps ? ps : g_g.psMask.Get(), g_g.dssOff.Get(), w, h);
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
    bool filled = false;             // forward pass (B9c): both slices got their content at the first bind or clear
    bool pending = false;            // B12: bound, its content goes in before the next draw (unless a clear of all)
    ComPtr<ID3D11DepthStencilView> slice1Dsv;  // depth: slice 1 only (the forward pass's copy-in)
    uint64_t lastFrame = 0;
    // B17: the game's texture is a two-slice array itself (clones::IsArray) and `arr` is it: slice 0 needs no copy in
    // or back.
    bool native = false;
    // B12, a multisampled color target of the forward pass: the game view's format (typed), slice 1 as a target (the
    // copy-in) and both slices for a shader (the corner out).
    DXGI_FORMAT viewFmt = DXGI_FORMAT_UNKNOWN;
    ComPtr<ID3D11RenderTargetView> slice1Rtv;
    ComPtr<ID3D11ShaderResourceView> colorSrv;
    // B12 step 5: slices of `arr` (3: the game's multisampled forward target as a three-slice array, mvr_ms_slices).
    UINT slices = 2;
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
    unsigned nativeUsed = 0;  // B17: targets that were the game's own arrays (nothing copied back)
    unsigned waitBinds = 0;   // binds and clears of other sizes before it
    std::string tailSeq;      // the tail's binds and draws (logged for the first passes with late binds)
    unsigned tailDraws = 0;   // draws since the latest tail bind
    bool lateBound = false;   // the latest tail bind holds an outer-size target
    unsigned lateDraws = 0;   // draws with an outer-size target bound in the tail (they would miss focus)
    bool failed = false;
    std::string why;
    double t0 = 0;
    // The forward pass (B9c): outer L's CPhase3Render in a forward test shot frame. Views per draw (VS and PS patched),
    // PS slot 3 and the focus copies of the screen textures in t21..t25 bound before each patched PS's draw.
    bool forward = false;
    bool vsPatched = false;          // the bound VS (VSSetShader hook)
    ID3D11Buffer* psSlot3 = nullptr;  // the game's binding of PS slot 3 at the start (referenced)
    unsigned psPatchedDraws = 0, psPlainDraws = 0, psOriginalDraws = 0, psUnknownDraws = 0, vsPlainDraws = 0;
    unsigned psCbRebinds = 0, srvMisses = 0, copyIns = 0;
    // B12: copies whose first use was a clear of all of them (no copy-in), multisampled targets (per-sample copies),
    // multisampled colors without a copy-in because the pass cleared its depth first (mvr_ms_color_skip).
    unsigned clearFills = 0, msTargets = 0, colorSkips = 0;
    bool depthCleared = false;  // a depth copy of this pass was cleared before its first use
    // B12 step 5: the slice view 1 draws into (renderTargetIndexOffset[1]): 2 in outer R's live forward pass on
    // three-slice arrays, else 1. sliceTargets: targets whose focus clone is that slice (no copy in or out).
    NvU32 viewSlice = 1;
    unsigned sliceTargets = 0, sliceClears = 0;
    // The forward pass also binds targets of other sizes (in the game a 1/2 size f9 with its own depth, run of
    // 2026-11-07 12:44): MVR pauses while they are bound (the game's views, one view) and goes on at the next bind with
    // an outer-size target.
    bool paused = false;
    unsigned pauses = 0, pausedDraws = 0, pausedClears = 0;
    // B9d: the live forward pass (every frame, either eye): its corners go into focus's clones at the end.
    bool live = false;
    // B9d run 5: the live track map of outer read in the focus window into a focus-size texture for view 1 (once per
    // forward pass, at the first draw whose PS reads it).
    bool liveTrackDone = false, liveTrackOk = false;
    // B13 (mvr_gbuffer_ps): the G-buffer pass's draws of patched PS; psSlot3 holds the game's PS slot 3 from the first.
    bool gbufPsBound = false;
    unsigned gbufPsDraws = 0, gbufPsKept = 0;  // with view 1's rows; kept as before (they read globalTexture0..4)
};
Pass g_p;

struct StashG {
    ID3D11Resource* orig = nullptr;  // the game's target as bound (only compared)
    ComPtr<ID3D11Texture2D> tex;     // the focus clone's layout
    D3D11_TEXTURE2D_DESC desc{};
    ComPtr<ID3D11DepthStencilView> dsv;
    bool depth = false;
    bool direct = false;  // B15: this frame's corner went straight into focus's clone (tex not written)
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
Win g_winPrev[2];  // the window of the last frame that had one (mvr_motion: the previous positions)
uint64_t g_fwdSkipFrame[2] = {};  // the frame in which the focus group skipped its own forward pass (mvr_motion)
struct FocusState {
    bool projected = false;
    uint64_t frame = 0;
    bool decided = false, skip = false;
    bool early = false;  // decided in outer's group, which also ran the group's SSAO (B9a)
    bool earlyLights = false;  // ... and its lights
    bool ssaoAgain = false;    // diagnosis (mode 4): the focus group runs its SSAO too
    bool refill = false;       // diagnosis (mode 5): the focus group gets the stash again at its G-buffer pass
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
    // B9a: focus's SSAO and lights in outer's group (runs, CPU, engine faults, the focus group's tasks skipped).
    uint64_t earlyRuns[2] = {}, earlyFaults = 0, earlySkipped[2] = {};
    double earlyMs[2] = {};
    uint64_t earlyWaits = 0, earlyWaited = 0;  // waits for the recording workers, of them non-zero
    double earlyWaitMs = 0, earlyWaitMax = 0;
    std::vector<Reason> earlyReasons;
    double passMs[2] = {};
    std::vector<Reason> reasons;
    // B13: G-buffer draws of patched PS with view 1's rows, kept as before; live track maps read for view 1.
    uint64_t gbufPsDraws = 0, gbufPsKept = 0, liveTracks = 0, liveTrackFails = 0;
    // B15: targets whose corner went straight into focus's clone, through the stash, direct writes that failed.
    uint64_t direct[2] = {}, viaStash[2] = {}, directFails = 0;
    // B17: targets of the MVR parts (0 G-buffer, 1 forward) that were the game's own arrays, that were copied back.
    uint64_t nativeT[2] = {}, copiedT[2] = {};
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

// The forward test (B9c, eye L): outer L's CPhase3Render with MVR in a shot frame, view 1's corner vs focus L's own
// forward pass. Its own state: the G-buffer comparison (g_shot) would make focus run its own G-buffer pass and stop the
// early SSAO and lights, which the forward pass needs in focus's clones.
struct FwdShot {
    bool armed = false, reported = false, outerDone = false, focusDone = false;
    int base = -1;
    uint64_t frame = 0;
    std::string why;  // why outer L's forward pass did not run with MVR
    Pass pass;        // its counters and rows (no references)
    double ms = 0;
    int n = 0;        // targets stashed
};
FwdShot g_fshot;
StashG g_fwd[kMaxTargets];  // the corners of slice 1 after the forward pass, in the focus size
int g_fwdN = 0;
ComPtr<ID3D11Texture2D> g_fwdDepthIn;  // focus's depth clone copied for a shader view (the copy-in)
struct SrvClone {
    ComPtr<ID3D11ShaderResourceView> game, clone;  // clone null: the texture has no focus clone
};
std::vector<SrvClone> g_srvClones;  // the forward pass's screen texture views on focus's clones
char g_fwdLabels[kMaxTargets][96] = {};
char g_fwdNames[2 * kMaxTargets][32] = {};
uint64_t g_earlyOk[2] = {};  // the frame of the eye's latest early SSAO and lights run without an engine fault

// MVR forward step 3 (B9d, mvr_pass.h): the live forward pass of each eye, the focus groups' skips, focus R's twins.
bool g_forwardOn = false;       // [debug] mvr_forward and its hotkey
int g_fwdDepth = 3;             // [debug] mvr_forward_depth and its hotkey (mvr_pass.h)
bool g_msColorSkip = true;      // [debug] mvr_ms_color_skip and its hotkey (mvr_pass.h)
bool g_msSlices = true;         // B12 step 5: [debug] mvr_ms_slices (restart only)
// Mode 3: per eye a conservative copy of the forward depth, bound in place of focus's depth clone while the focus
// group's velocity pass (an R16G16 target at the focus size) draws.
struct VelDepth {
    ID3D11Resource* orig = nullptr;  // the engine's depth texture (only compared)
    ComPtr<ID3D11Texture2D> tex;
    uint64_t frame = 0;
};
VelDepth g_vel[2];
int g_curFocus = -1;  // the focus group whose clones are in (FocusGroupIn)
uintptr_t g_phase3BuildVt = 0;  // focus's CPhase3BuildCommandThread (quad, by class name)
uint64_t g_fwdOk[2] = {};       // the frame whose forward pass of the eye put view 1 into focus's clones
std::atomic<uint64_t> g_buildRan[2] = {};  // the frame in which focus's CPhase3BuildCommandThread ran (then its
                                            // CPhase3Render runs too: the pair records and replays its buffers)
bool g_track = false;           // an early run: the focus clones its tasks bind or clear are noted
// The engine textures whose focus clones the early runs or the forward passes write (only compared). Focus L and R
// share the focus clones, and outer R's group writes focus R's results before focus L's group reads focus L's: each
// of these clones needs focus R's own copy (a twin, clones::WantTwin) before focus R's work goes into outer R's group.
constexpr size_t kMaxTwins = 48;
std::vector<ID3D11Resource*> g_twinOrigs;
// A clear of a target of another size in the live forward pass (repeated on its focus clone at the end).
struct OtherClear {
    ID3D11Resource* orig = nullptr;  // only compared
    bool depth = false;
    D3D11_RENDER_TARGET_VIEW_DESC rd{};
    D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
    float col[4] = {};
    UINT flags = 0;
    float d = 0;
    UINT8 s = 0;
};
std::vector<OtherClear> g_otherClears;
// The module's views on focus's clones (the forward pass's depth target, the repeated clears).
struct CloneView {
    ID3D11Resource* tex = nullptr;  // only compared (the view holds a reference)
    int kind = 0;                   // 0 RTV, 1 DSV, 2 SRV (B12)
    uint8_t desc[32] = {};
    ComPtr<ID3D11View> v;
};
std::vector<CloneView> g_cloneViews;
// The live forward pass per stats interval and for the session (in Stats would make it longer still).
struct FwdStats {
    uint64_t passes[2] = {}, ok[2] = {}, skipped[2] = {}, buildSkipped[2] = {}, own[2] = {};
    uint64_t draws = 0, otherClears = 0, otherClearMisses = 0;
    uint64_t velBinds = 0;  // mode 3: velocity pass binds given the conservative depth
    // B11: velocity passes that got outer's vectors and skipped their draws, passes that drew (with reasons).
    uint64_t velFilled[2] = {}, velDrawn[2] = {}, velSkippedDraws = 0;
    std::vector<Reason> velReasons;
    uint64_t liveTracks = 0, liveTrackFails = 0;  // live track maps read in the focus window for view 1
    uint64_t copyIns = 0, clearFills = 0, msTargets = 0, colorSkips = 0;  // B12 (Pass)
    uint64_t sliceTargets[2] = {}, sliceClears = 0;  // B12 step 5 (Pass), per eye
    double ms[2] = {};
    std::vector<Reason> reasons;
};
FwdStats g_fs, g_ftot;
std::string g_lastFwdFail;
struct TestLive {
    unsigned passes = 0, ok = 0, focusSkipped = 0, buildSkipped = 0, otherClears = 0, otherClearMisses = 0;
    unsigned velBinds = 0, velFilled = 0, velSkippedDraws = 0, velDrawn = 0;
    unsigned native = 0, copiedBack = 0;  // B17: the last live pass's targets that were the game's arrays, copied back
    unsigned copyIns = 0, clearFills = 0, msTargets = 0, colorSkips = 0;  // B12: the last live pass's
    unsigned sliceTargets = 0, sliceClears = 0;                           // B12 step 5: the last live pass's
    std::string why;
};
TestLive g_testLive;
// B11 (mvr_pass.h): the focus velocity pass of the frame per eye, and the eye whose velocity pass is bound now.
enum VelState { VelUndecided, VelFilled, VelDraws };
struct VelSkip {
    uint64_t frame = 0;
    VelState state = VelUndecided;
};
VelSkip g_velSkip[2];
std::atomic<int> g_velBound{-1};
bool g_velSkipOn = true;  // [debug] mvr_velocity_skip and its hotkey
VelocityFillFn g_velFill = nullptr;
bool g_gbufPsOn = true;  // B13: [debug] mvr_gbuffer_ps and its hotkey
bool g_directOn = true;  // B15: [debug] mvr_direct_extract and its hotkey
// B17: [debug] mvr_native_targets and its hotkey; the smoke's stand-ins for clones::IsArray and clones::WantArray.
bool g_nativeOn = true;
uint32_t g_cloneGen = 0;  // clones::Generation() last seen
NativeFn g_nativeFn = nullptr;
WantArrayFn g_wantFn = nullptr;
bool IsNative(ID3D11Resource* r) { return g_nativeFn ? g_nativeFn(r) : clones::IsArray(r); }
void WantNative(ID3D11Resource* r, UINT slices) {
    if (g_wantFn)
        g_wantFn(r, slices);
    else if (!g_cloneFn)
        clones::WantArray(r, slices);
}

// Jitter diagnosis (B9c run 2: view 1 about 0.27 px off focus's own forward pass in x, the sign changing between
// frames): in a forward test shot frame the game's camera constants at the first draw of each eye group's
// CPhase1Render and CPhase3Render: VS slot 12 (COMMON_BUFFER2, capture B2): projMatrix at byte 128,
// screenSpaceDimensions 384, viewportPixelDimensions 464, cameraJitter 944 (61 registers).
constexpr UINT kCamSlot = 12, kCamBytes = 976, kCamProj = 128, kCamSsd = 384, kCamVpd = 464, kCamJitter = 944;
struct Probe {
    bool on = false, done = false;
    int group = -1;
    uint8_t* task = nullptr;
    const char* name = "";
};
Probe g_probe;
const char* const kGroupName[4] = {"outer L", "outer R", "focus L", "focus R"};

void FwdRefused(const char* why) {
    if (g_fshot.armed && g_fshot.why.empty() && why) g_fshot.why = why;
}

// Test (smoke) snapshot of the latest pass.
TestState g_test;

bool SetViews(ID3D11DeviceContext* c, NvU32 views) {
    NV_MULTIVIEW_PARAMS p = {};
    p.version = NV_MULTIVIEW_PARAMS_VER;
    p.numViews = views;
    for (NvU32 i = 0; i < 4; ++i) p.renderTargetIndexOffset[i] = i;
    p.renderTargetIndexOffset[1] = g_p.viewSlice;  // B12 step 5: 2 in outer R's pass on three-slice arrays
    const NvAPI_Status st = NvAPI_D3D_SetMultiViewMode(c, &p);
    if (st == NVAPI_OK) return true;
    if (!g_p.viewsFailures++ && !g_st.viewsFailures)
        Log("MVR G-buffer: NvAPI_D3D_SetMultiViewMode(%u views) %s", (unsigned)views, NvErr(st).c_str());
    return false;
}

void ReleasePassRefs(Pass& p) {
    if (p.slot3) p.slot3->Release();
    p.slot3 = nullptr;
    if (p.psSlot3) p.psSlot3->Release();
    p.psSlot3 = nullptr;
}

void ResetLiveTrack();
void ResetMsScratch();

void ReleaseAll() {
    ReleasePassRefs(g_p);
    g_p = Pass{};
    g_copies.clear();
    g_views.clear();
    for (Stash& s : g_stash) s = Stash{};
    g_shot = Shot{};
    g_fshot = FwdShot{};
    for (StashG& g : g_fwd) g = StashG{};
    g_fwdN = 0;
    g_fwdDepthIn.Reset();
    ResetMsScratch();
    g_srvClones.clear();
    g_twinOrigs.clear();
    g_otherClears.clear();
    g_cloneViews.clear();
    for (VelDepth& v : g_vel) v = VelDepth{};
    ResetLiveTrack();
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
    if (!ok) return -1;
    // B17: the game's own two-slice array (clones.h) is the copy itself unless mvr_native_targets is off (then a copy
    // like before, with slice 0 copied in and back: the A/B of the copies' cost).
    const bool isArray = (d.ArraySize == 2 || d.ArraySize == 3) && IsNative(res);  // B12 step 5: three slices
    if (d.ArraySize != 1 && !isArray) return -1;
    const bool native = isArray && g_nativeOn;
    int slot = -1;
    for (int k = 0; k < (int)g_copies.size(); ++k) {
        Copy& cp = g_copies[k];
        if (cp.orig != res) continue;
        if (SameLayout(cp.desc, d) && cp.native == native) {
            cp.used = true;
            cp.lastFrame = g_frame;
            return k;
        }
        slot = k;  // a new texture at the address of a released one (or the other B17 mode)
        break;
    }
    if (slot < 0) {
        for (int k = 0; k < (int)g_copies.size() && slot < 0; ++k)
            if (!g_copies[k].arr) slot = k;  // released at a frame start
    }
    if (slot < 0 && (int)g_copies.size() >= kMaxCopies) {
        for (int k = 0; k < (int)g_copies.size() && slot < 0; ++k)
            if (g_copies[k].lastFrame + 2 < g_frame) slot = k;
        if (slot < 0) return -1;
    }
    // The G-buffer's targets at the outer size become the game's own arrays at the next frame start (B17), and so do
    // the forward pass's multisampled ones (B12: a copy of slice 0 in and back is 8 times the data with MSAA 8x; the
    // single-sample forward color stays a copy, DLAA's inputs would need a copy of an array).
    // B12 step 5 (mvr_ms_slices): the multisampled ones with three slices (focus R's view 1 in slice 2).
    if (!isArray && g_nativeOn && t_pass && d.Width == g_p.w && d.Height == g_p.h && d.MipLevels == 1 &&
        (g_p.forward ? d.SampleDesc.Count > 1 : d.SampleDesc.Count == 1))
        WantNative(res, g_p.forward && g_msSlices ? 3 : 2);
    Copy cp;
    cp.orig = res;
    cp.desc = d;
    cp.depth = (d.BindFlags & D3D11_BIND_DEPTH_STENCIL) != 0;
    cp.native = native;
    cp.slices = native ? d.ArraySize : 2;
    DXGI_FORMAT dsvF, srvF;
    DepthFormats(d.Format, dsvF, srvF);
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    if (!dev) return -1;
    if (native) {
        if (FAILED(res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&cp.arr))) return -1;
        if (!(d.BindFlags & D3D11_BIND_SHADER_RESOURCE)) srvF = DXGI_FORMAT_UNKNOWN;  // no depth view: no stash
    } else {
        D3D11_TEXTURE2D_DESC a = d;
        a.ArraySize = 2;
        a.MiscFlags &= ~(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                         D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE |
                         D3D11_RESOURCE_MISC_TEXTURECUBE);
        if (cp.depth && srvF != DXGI_FORMAT_UNKNOWN) a.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev->CreateTexture2D(&a, nullptr, &cp.arr))) return -1;
    }
    if (cp.depth && srvF != DXGI_FORMAT_UNKNOWN) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = srvF;
        if (d.SampleDesc.Count > 1) {
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
            sv.Texture2DMSArray = {0, cp.slices};
        } else {
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            sv.Texture2DArray = {0, 1, 0, cp.slices};
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
        d.Texture2DArray = {mip, 0, k >= 0 ? g_copies[k].slices : 2};
    } else if (d.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DMS) {
        d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, k >= 0 ? g_copies[k].slices : 2};
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
        d.Texture2DArray = {mip, 0, k >= 0 ? g_copies[k].slices : 2};
    } else if (d.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DMS) {
        d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {0, k >= 0 ? g_copies[k].slices : 2};
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

// B12 step 5: the slice view 1 of this pass draws into for copy cp: 2 in outer R's live forward pass on a three-slice
// array (slice 1 holds focus L's forward until focus L's group), else 1.
NvU32 SliceFor(const Copy& cp) { return g_p.live && g_p.eye == 1 && cp.slices >= 3 ? 2 : 1; }
// The slice of cp's array that is focus's clone for the pass's eye (0: a clone of its own; only three-slice arrays).
int FocusSliceOfCopy(const Copy& cp) {
    return cp.native && cp.slices >= 3 && g_p.eye >= 0 ? FocusSliceOf(cp.orig, g_p.eye) : 0;
}

// ---------------------------------------------------------------- B9d: focus clones written, focus R's twins, views

// Bytes per pixel of the formats the eye targets use (4 for the rest; only for the logged size).
double BytesPerPixel(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G8X24_TYPELESS: return 8;
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: return 2;
    case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R8_UNORM: return 1;
    default: return 4;
    }
}

void FwdReason(int eye, const char* why) {
    char b[200];
    snprintf(b, sizeof(b), "%s %s", kEye[eye], why);
    AddReason(g_fs.reasons, b);
    AddReason(g_ftot.reasons, b);
}

// A focus clone written by an early run or a forward pass, by its engine texture: it gets focus R's twin.
void NoteOrig(ID3D11Resource* orig) {
    if (!orig || !g_forwardOn) return;  // B8's fills alone need no twins
    for (ID3D11Resource* o : g_twinOrigs)
        if (o == orig) return;
    if (g_twinOrigs.size() >= kMaxTwins) return;
    g_twinOrigs.push_back(orig);
    if (!g_cloneFn) clones::WantTwin(orig);
}

// Every focus clone noted has focus R's twin (asked again for the missing ones; textures without a focus clone any
// more, after a list change, are dropped). Eye L's work notes them, so focus R's early run and forward pass wait one
// frame after a list change. The smoke's fake clones have none.
bool TwinsReady() {
    if (g_cloneFn) return true;
    bool ready = true;
    for (size_t i = 0; i < g_twinOrigs.size();) {
        ID3D11Resource* o = g_twinOrigs[i];
        if (!clones::CloneOf(o, clones::Focus)) {
            g_twinOrigs.erase(g_twinOrigs.begin() + i);
            continue;
        }
        if (!clones::HasTwin(o)) {
            clones::WantTwin(o);
            ready = false;
        }
        ++i;
    }
    return ready && !g_twinOrigs.empty();
}

// A view bound or cleared during an early run (focus's clones are in): its texture when it is a clone.
void NoteWritten(ID3D11View* v) {
    if (!v || g_cloneFn) return;  // the smoke's fake clones are noted where the module writes them
    ComPtr<ID3D11Resource> r;
    v->GetResource(&r);
    ID3D11Resource* orig = (ID3D11Resource*)clones::Original(r.Get());
    if (orig && orig != r.Get()) NoteOrig(orig);
}

// The module's RTV (kind 0) or DSV (kind 1) with `desc` on a focus clone (kept; null on failure).
ID3D11View* CloneViewOf(ID3D11Texture2D* t, int kind, const void* desc, size_t n) {
    if (!t || n > sizeof(CloneView::desc)) return nullptr;
    for (const CloneView& e : g_cloneViews)
        if (e.tex == t && e.kind == kind && !memcmp(e.desc, desc, n)) return e.v.Get();
    if (g_cloneViews.size() >= 64) g_cloneViews.clear();  // both eyes' clones
    CloneView e;
    e.tex = t;
    e.kind = kind;
    memcpy(e.desc, desc, n);
    HRESULT hr = E_FAIL;
    if (kind == 0) {
        ComPtr<ID3D11RenderTargetView> v;
        hr = g_g.dev->CreateRenderTargetView(t, (const D3D11_RENDER_TARGET_VIEW_DESC*)desc, &v);
        e.v = v;
    } else if (kind == 2) {
        ComPtr<ID3D11ShaderResourceView> v;
        hr = g_g.dev->CreateShaderResourceView(t, (const D3D11_SHADER_RESOURCE_VIEW_DESC*)desc, &v);
        e.v = v;
    } else {
        ComPtr<ID3D11DepthStencilView> v;
        hr = g_g.dev->CreateDepthStencilView(t, (const D3D11_DEPTH_STENCIL_VIEW_DESC*)desc, &v);
        e.v = v;
    }
    if (FAILED(hr) || !e.v) return nullptr;
    g_cloneViews.push_back(e);
    return g_cloneViews.back().v.Get();
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

// PS slot 3 of the forward pass (mvr_ps_patch.h): row 0 = S and O of the screen inputs from view 1's corner map in M
// (x_c = S.x x + o.x w, y_c = S.y y + o.y w; the screen UV scale ssd.xy of the game's VS taken as 1, B2), row 1 = the
// UV scale from the outer-size slice to the focus-size textures, row 2.x = the magic value that turns the view on.
void PsRows(const Pass& p, float r[4][4]) {
    const float sx = p.rows[0][0], ox = p.rows[0][3], sy = p.rows[1][1], oy = p.rows[1][3];
    const float rows[4][4] = {{sx, sy, 0.5f * (1 + ox - sx), 0.5f * (1 - oy - sy)},
                              {(float)p.w / p.fw, (float)p.h / p.fh, 1, 1},
                              {},
                              {}};
    memcpy(r, rows, sizeof(rows));
    const uint32_t magic = dxbc::kMvrPsMagic;
    memcpy(&r[2][0], &magic, sizeof(magic));
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
    if (p.forward || g_gbufPsOn) {
        float r[4][4];
        PsRows(p, r);
        c->UpdateSubresource(g_g.psRowsCb.Get(), 0, nullptr, r, 0, 0);
    }
    p.mFilled = true;
}

// ---------------------------------------------------------------- outer's pass

void PassFailed(Pass& p, const char* why) {
    if (!p.failed) p.why = why;
    p.failed = true;
}

// The forward pass (B9c) draws over what the earlier tasks left, so each copy gets it at its first bind or clear:
// slice 0 = the game's texture, the corner of slice 1 = focus's clone (focus L's G-buffer from the stash, its SSAO and
// lights from the early run; the depth through PSDepthIn, D3D11 copies no part of a depth texture).
// B12: a shader view (Texture2DMS, `fmt`) of a multisampled focus clone: on the clone itself when it has the bind,
// else on a copy of it (a multisampled texture keeps its render target or depth bind).
struct MsScratch {
    ComPtr<ID3D11Texture2D> tex;
    D3D11_TEXTURE2D_DESC desc{};
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    ComPtr<ID3D11ShaderResourceView> srv;
};
std::vector<MsScratch> g_msScratch;
void ResetMsScratch() { g_msScratch.clear(); }
ID3D11ShaderResourceView* ReadableMs(ID3D11DeviceContext* c, ID3D11Texture2D* t, DXGI_FORMAT fmt) {
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = fmt;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
    if (d.BindFlags & D3D11_BIND_SHADER_RESOURCE)
        return static_cast<ID3D11ShaderResourceView*>(CloneViewOf(t, 2, &sv, sizeof(sv)));
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE |
                  ((d.BindFlags & D3D11_BIND_DEPTH_STENCIL) ? D3D11_BIND_DEPTH_STENCIL : D3D11_BIND_RENDER_TARGET);
    d.MiscFlags = 0;
    d.CPUAccessFlags = 0;
    d.Usage = D3D11_USAGE_DEFAULT;
    MsScratch* s = nullptr;
    for (MsScratch& e : g_msScratch)
        if (SameLayout(e.desc, d) && e.fmt == fmt) s = &e;
    if (!s) {
        if (g_msScratch.size() >= 4) g_msScratch.clear();
        MsScratch e;
        e.desc = d;
        e.fmt = fmt;
        if (FAILED(g_g.dev->CreateTexture2D(&d, nullptr, &e.tex)) ||
            FAILED(g_g.dev->CreateShaderResourceView(e.tex.Get(), &sv, &e.srv)))
            return nullptr;
        g_msScratch.push_back(std::move(e));
        s = &g_msScratch.back();
    }
    Internal in;
    c->CopySubresourceRegion(s->tex.Get(), 0, 0, 0, 0, t, 0, nullptr);
    return s->srv.Get();
}

bool DepthIn(ID3D11DeviceContext* c, Copy& cp, ID3D11Texture2D* clone) {
    DXGI_FORMAT dsvF, srvF;
    DepthFormats(cp.desc.Format, dsvF, srvF);
    if (dsvF == DXGI_FORMAT_UNKNOWN || srvF == DXGI_FORMAT_UNKNOWN) return false;
    const bool ms = cp.desc.SampleDesc.Count > 1;
    if (!cp.slice1Dsv) {
        D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = dsvF;
        if (ms) {
            dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY;
            dv.Texture2DMSArray = {1, 1};
        } else {
            dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
            dv.Texture2DArray = {0, 1, 1};
        }
        if (FAILED(g_g.dev->CreateDepthStencilView(cp.arr.Get(), &dv, &cp.slice1Dsv))) return false;
    }
    if (ms) {  // B12: per sample
        ID3D11ShaderResourceView* v = ReadableMs(c, clone, srvF);
        if (!v) return false;
        Internal in;
        Saved s;
        s.Save(c);
        c->OMSetRenderTargets(0, nullptr, cp.slice1Dsv.Get());
        c->PSSetShaderResources(0, 1, &v);
        FullScreen(c, g_g.psDepthInMs.Get(), g_g.dssWrite.Get(), g_p.fw, g_p.fh);
        ID3D11ShaderResourceView* none = nullptr;
        c->PSSetShaderResources(0, 1, &none);
        s.Restore(c);
        return true;
    }
    D3D11_TEXTURE2D_DESC cd{};
    clone->GetDesc(&cd);
    cd.MipLevels = 1;
    cd.ArraySize = 1;
    cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    cd.MiscFlags = 0;
    cd.CPUAccessFlags = 0;
    cd.Usage = D3D11_USAGE_DEFAULT;
    D3D11_TEXTURE2D_DESC have{};
    if (g_fwdDepthIn) g_fwdDepthIn->GetDesc(&have);
    if (!g_fwdDepthIn || !SameLayout(have, cd)) {
        g_fwdDepthIn.Reset();
        if (FAILED(g_g.dev->CreateTexture2D(&cd, nullptr, &g_fwdDepthIn))) return false;
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = srvF;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D = {0, 1};
    ComPtr<ID3D11ShaderResourceView> srv;
    if (FAILED(g_g.dev->CreateShaderResourceView(g_fwdDepthIn.Get(), &sv, &srv))) return false;
    Internal in;
    c->CopySubresourceRegion(g_fwdDepthIn.Get(), 0, 0, 0, 0, clone, 0, nullptr);
    Saved s;
    s.Save(c);
    c->OMSetRenderTargets(0, nullptr, cp.slice1Dsv.Get());
    ID3D11ShaderResourceView* v = srv.Get();
    c->PSSetShaderResources(0, 1, &v);
    FullScreen(c, g_g.psDepthIn.Get(), g_g.dssWrite.Get(), g_p.fw, g_p.fh);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    s.Restore(c);
    return true;
}

// B12: focus's multisampled color clone into the corner of slice 1, sample by sample.
bool ColorInMs(ID3D11DeviceContext* c, Copy& cp, ID3D11Texture2D* clone) {
    if (cp.viewFmt == DXGI_FORMAT_UNKNOWN) return false;
    if (!cp.slice1Rtv) {
        D3D11_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = cp.viewFmt;
        rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY;
        rv.Texture2DMSArray = {1, 1};
        if (FAILED(g_g.dev->CreateRenderTargetView(cp.arr.Get(), &rv, &cp.slice1Rtv))) return false;
    }
    ID3D11ShaderResourceView* v = ReadableMs(c, clone, cp.viewFmt);
    if (!v) return false;
    ColorPassMs(c, v, cp.slice1Rtv.Get(), g_p.fw, g_p.fh, g_g.psColorInMs.Get());
    return true;
}

// The copy behind a view of a copy (-1: none), its color view's format noted (B12).
int CopyOfView(ID3D11View* v) {
    int k = -1;
    if (!v || !FindView(v, &k) || k < 0 || k >= (int)g_copies.size()) return -1;
    Copy& cp = g_copies[k];
    if (!cp.depth && cp.viewFmt == DXGI_FORMAT_UNKNOWN) {
        ComPtr<ID3D11RenderTargetView> rv;
        if (SUCCEEDED(v->QueryInterface(IID_PPV_ARGS(&rv)))) {
            D3D11_RENDER_TARGET_VIEW_DESC d{};
            rv->GetDesc(&d);
            cp.viewFmt = d.Format;
        }
    }
    return k;
}

// Copy k's content: slice 0 = the game's texture, the corner of slice 1 = focus's clone (once per copy and pass).
// `cleared`: the game clears all of it now (B12: then neither slice needs its content first).
void FillCopyAt(ID3D11DeviceContext* c, int k, bool cleared) {
    Pass& p = g_p;
    Copy& cp = g_copies[k];
    cp.pending = false;
    if (cp.filled) return;
    cp.filled = true;
    const bool ms = cp.desc.SampleDesc.Count > 1;
    if (ms) ++p.msTargets;
    const bool skip = cleared && cp.desc.MipLevels == 1;
    clones::RawScope raw;  // outer's group: the bind fix would copy the original in place of focus's clone
    const NvU32 views = p.views;
    if (views != 1) SetViews(c, 1);  // the depth copy-in draws
    if (!cp.native && !skip) {  // B17: a native copy's slice 0 is the game's texture
        Internal in;
        for (UINT m = 0; m < cp.desc.MipLevels; ++m)
            c->CopySubresourceRegion(cp.arr.Get(), D3D11CalcSubresource(m, 0, cp.desc.MipLevels), 0, 0, 0, cp.orig,
                                     m, nullptr);
    }
    const bool outerSize = cp.desc.Width == p.w && cp.desc.Height == p.h;
    ID3D11Texture2D* clone = outerSize ? FocusClone(cp.orig) : nullptr;
    D3D11_TEXTURE2D_DESC cd{};
    if (clone) clone->GetDesc(&cd);
    const int fs = outerSize ? FocusSliceOfCopy(cp) : 0;
    if (!outerSize) {
        PassFailed(p, "a forward target not at the outer size");
    } else if (fs) {
        // B12 step 5: focus's clone is this slice of the array, so focus's content is in its corner already.
        if (fs != (int)SliceFor(cp)) {
            PassFailed(p, "focus's clone of a forward target is another slice than view 1 draws into");
        } else {
            ++p.sliceTargets;
            if (skip) {
                ++p.clearFills;
                if (cp.depth) p.depthCleared = true;
            }
        }
    } else if (SliceFor(cp) != 1) {
        PassFailed(p, "view 1 in slice 2 of a forward target without focus R's slice clone");
    } else if (!clone) {
        PassFailed(p, "a forward target without a focus clone");
    } else if (cd.Width != p.fw || cd.Height != p.fh || cd.Format != cp.desc.Format ||
               cd.SampleDesc.Count != cp.desc.SampleDesc.Count || cd.SampleDesc.Quality != cp.desc.SampleDesc.Quality) {
        PassFailed(p, "a forward target's focus clone of another size or format");
    } else if (skip) {
        ++p.clearFills;
        if (cp.depth) p.depthCleared = true;
    } else if (!cp.depth && ms && p.depthCleared && g_msColorSkip) {
        // B12: a multisampled forward pass that starts from a cleared depth shades its color from scratch (in AMS2 the
        // opaque sky dome covers every sample first), so focus's old color in slice 1 is not needed.
        ++p.colorSkips;
    } else if (!cp.depth) {
        if (ms) {
            if (ColorInMs(c, cp, clone))
                ++p.copyIns;
            else
                PassFailed(p, "focus's multisampled color could not go into slice 1");
        } else {
            Internal in;
            const D3D11_BOX box{0, 0, 0, p.fw, p.fh, 1};
            c->CopySubresourceRegion(cp.arr.Get(), D3D11CalcSubresource(0, 1, cp.desc.MipLevels), 0, 0, 0, clone, 0,
                                     &box);
            ++p.copyIns;
        }
    } else if (DepthIn(c, cp, clone)) {
        ++p.copyIns;
    } else {
        PassFailed(p, "focus's depth could not go into slice 1");
    }
    if (views != 1) SetViews(c, views);
}

// A clear of the copy behind `v` (B12: one of all of it gives no copy-in).
void FillCopy(ID3D11DeviceContext* c, ID3D11View* v, bool cleared) {
    const int k = CopyOfView(v);
    if (k >= 0) FillCopyAt(c, k, cleared);
}

// A bind: the copy gets its content before the next draw (B12: the game often binds and then clears).
void MarkPending(ID3D11View* v) {
    const int k = CopyOfView(v);
    if (k >= 0 && !g_copies[k].filled) g_copies[k].pending = true;
}

// Before a draw of the forward pass and at the end of its MVR part.
void FlushPending(ID3D11DeviceContext* c) {
    for (int k = 0; k < (int)g_copies.size(); ++k)
        if (g_copies[k].pending) FillCopyAt(c, k, false);
}

// The live track map for view 1 (B9d run 5): outer's map (bound by the game, outer size) read in the eye's focus window
// into a focus-size texture once per forward pass (B13: and once per G-buffer pass). Focus's own map is drawn only
// later, in its group. Null: none (the draw counts as a screen texture without a copy).
struct LiveTrackMap {
    ComPtr<ID3D11Texture2D> tex;
    D3D11_TEXTURE2D_DESC desc{};
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11RenderTargetView> rtv;
};
LiveTrackMap g_liveTrack;
void ResetLiveTrack() { g_liveTrack = LiveTrackMap{}; }

ID3D11ShaderResourceView* LiveTrackSrv(ID3D11DeviceContext* c, ID3D11ShaderResourceView* g) {
    Pass& p = g_p;
    if (p.liveTrackDone) return p.liveTrackOk ? g_liveTrack.srv.Get() : nullptr;
    p.liveTrackDone = true;
    auto fail = [&]() -> ID3D11ShaderResourceView* {
        ++(p.forward ? g_fs.liveTrackFails : g_st.liveTrackFails);
        ++(p.forward ? g_ftot.liveTrackFails : g_tot.liveTrackFails);
        return nullptr;
    };
    ComPtr<ID3D11Resource> r;
    g->GetResource(&r);
    bool ok = false;
    const D3D11_TEXTURE2D_DESC d = DescOf(r.Get(), &ok);
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    g->GetDesc(&sv);
    if (!ok || d.Width != p.w || d.Height != p.h || d.ArraySize != 1 || d.SampleDesc.Count != 1 ||
        sv.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D)
        return fail();
    D3D11_TEXTURE2D_DESC want = d;
    want.Width = p.fw;
    want.Height = p.fh;
    want.MipLevels = 1;
    want.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    want.MiscFlags = 0;
    want.CPUAccessFlags = 0;
    want.Usage = D3D11_USAGE_DEFAULT;
    LiveTrackMap& m = g_liveTrack;
    if (!m.tex || !SameLayout(m.desc, want) || m.fmt != sv.Format) {
        m = LiveTrackMap{};
        D3D11_SHADER_RESOURCE_VIEW_DESC s2{};
        s2.Format = sv.Format;
        s2.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        s2.Texture2D = {0, 1};
        D3D11_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = sv.Format;
        rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        if (FAILED(g_g.dev->CreateTexture2D(&want, nullptr, &m.tex)) ||
            FAILED(g_g.dev->CreateShaderResourceView(m.tex.Get(), &s2, &m.srv)) ||
            FAILED(g_g.dev->CreateRenderTargetView(m.tex.Get(), &rv, &m.rtv))) {
            m = LiveTrackMap{};
            return fail();
        }
        m.desc = want;
        m.fmt = sv.Format;
    }
    const Win& w = g_win[p.eye];
    const float origin[2] = {(float)((1 - 1 / w.sx - w.ox / w.sx) / 2), (float)((1 - 1 / w.sy + w.oy / w.sy) / 2)};
    const float scale[2] = {(float)(1 / w.sx), (float)(1 / w.sy)};
    const NvU32 views = p.views;
    if (views != 1) SetViews(c, 1);
    MaskPass(c, g, m.rtv.Get(), p.fw, p.fh, origin, scale, g_g.psBlit.Get());
    if (views != 1) SetViews(c, views);
    p.liveTrackOk = true;
    ++(p.forward ? g_fs.liveTracks : g_st.liveTracks);
    ++(p.forward ? g_ftot.liveTracks : g_tot.liveTracks);
    return m.srv.Get();
}

// The view of a screen texture on its focus clone, with the game view's description (null: no clone).
ID3D11ShaderResourceView* CloneSrv(ID3D11ShaderResourceView* v) {
    for (const SrvClone& e : g_srvClones)
        if (e.game.Get() == v) return e.clone.Get();
    SrvClone e;
    e.game = v;
    ComPtr<ID3D11Resource> r;
    v->GetResource(&r);
    // B12 step 5: a slice clone's array has no view of focus's own here (the draw counts as a screen texture miss).
    if (ID3D11Texture2D* clone = FocusSliceOf(r.Get(), g_p.eye) ? nullptr : FocusClone(r.Get())) {
        D3D11_SHADER_RESOURCE_VIEW_DESC d{};
        v->GetDesc(&d);
        if (FAILED(g_g.dev->CreateShaderResourceView(clone, &d, &e.clone))) e.clone.Reset();
    }
    if (g_srvClones.size() < 64) g_srvClones.push_back(e);
    return e.clone.Get();
}

// Before each draw of the forward pass: two views when the VS and the PS are right for view 1 (VS patched; PS
// patched or not needing it), else one; a patched PS gets the rows in slot 3 and focus's copies of its screen textures.
void ForwardDraw(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    if (p.paused) {
        ++p.pausedDraws;
        ++p.draws;
        ++p.plainDraws;
        if (p.views != 1 && SetViews(c, 1)) p.views = 1;
        return;
    }
    ID3D11PixelShader* ps = nullptr;
    c->PSGetShader(&ps, nullptr, nullptr);
    const mvrps::Info info = mvrps::InfoOf(ps);
    if (ps) ps->Release();
    const bool psOk = info.kind == mvrps::Kind::Patched || info.kind == mvrps::Kind::NotNeeded;
    if (info.kind == mvrps::Kind::Patched && p.vsPatched) {
        ++p.psPatchedDraws;
        ID3D11Buffer* cur = nullptr;
        c->PSGetConstantBuffers(mvrps::kCbSlot, 1, &cur);
        ID3D11Buffer* mine = g_g.psRowsCb.Get();
        if (cur != mine) {
            Internal in;
            c->PSSetConstantBuffers(mvrps::kCbSlot, 1, &mine);
            ++p.psCbRebinds;
        }
        if (cur) cur->Release();
        ID3D11ShaderResourceView* tex[mvrps::kTextures] = {};
        for (unsigned k = 0; k < mvrps::kTextures; ++k) {
            if (info.slot[k] == mvrps::kNoSlot) continue;
            ID3D11ShaderResourceView* g = nullptr;
            c->PSGetShaderResources(info.slot[k], 1, &g);
            if (!g) continue;
            tex[k] = k == mvrps::kLiveTrack ? LiveTrackSrv(c, g) : CloneSrv(g);
            if (!tex[k]) ++p.srvMisses;
            g->Release();
        }
        Internal in;
        clones::RawScope raw;
        c->PSSetShaderResources(mvrps::kTexBase, mvrps::kTextures, tex);
    }
    if (!p.vsPatched)
        ++p.vsPlainDraws;
    else if (!psOk)
        ++(info.kind == mvrps::Kind::Original ? p.psOriginalDraws : p.psUnknownDraws);
    const NvU32 want = p.vsPatched && psOk ? 2 : 1;
    if (want != p.views && SetViews(c, want)) p.views = want;
    ++p.draws;
    if (p.views != 2) ++p.plainDraws;
}

// B13: before each draw of the G-buffer pass with two views: a patched PS gets PS slot 3 (view 1's rows, the magic
// value) and outer's live track map in the focus window. One that also reads globalTexture0..4 gets the game's slot 3
// (no magic: view 1 draws it as before); the G-buffer pass has no focus copies of them (none in the capture, B2).
// With one view the patched PS draw as the original whatever slot 3 holds.
void GbufDraw(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    if (p.views != 2) return;
    ID3D11PixelShader* ps = nullptr;
    c->PSGetShader(&ps, nullptr, nullptr);
    const mvrps::Info info = mvrps::InfoOf(ps);
    if (ps) ps->Release();
    if (info.kind != mvrps::Kind::Patched) return;
    if (!p.gbufPsBound) {
        c->PSGetConstantBuffers(mvrps::kCbSlot, 1, &p.psSlot3);
        p.gbufPsBound = true;
    }
    bool screen = false;
    for (unsigned k = 0; k < mvrps::kLiveTrack; ++k) screen |= info.slot[k] != mvrps::kNoSlot;
    ID3D11Buffer* want = screen ? p.psSlot3 : g_g.psRowsCb.Get();
    ID3D11Buffer* cur = nullptr;
    c->PSGetConstantBuffers(mvrps::kCbSlot, 1, &cur);
    if (cur != want) {
        Internal in;
        c->PSSetConstantBuffers(mvrps::kCbSlot, 1, &want);
        ++p.psCbRebinds;
    }
    if (cur) cur->Release();
    if (screen) {
        ++p.gbufPsKept;
        return;
    }
    ++p.gbufPsDraws;
    ID3D11ShaderResourceView* map = nullptr;
    if (info.slot[mvrps::kLiveTrack] != mvrps::kNoSlot) {
        ID3D11ShaderResourceView* g = nullptr;
        c->PSGetShaderResources(info.slot[mvrps::kLiveTrack], 1, &g);
        if (g) {
            map = LiveTrackSrv(c, g);
            if (!map) ++p.srvMisses;
            g->Release();
        }
    }
    Internal in;
    clones::RawScope raw;
    c->PSSetShaderResources(mvrps::kTexBase + mvrps::kLiveTrack, 1, &map);
}

// A constant buffer range the next draw reads (VS `slot` from its binding offset), read back at once (shot frame only:
// the read-back waits for the GPU). False: nothing bound. `how` says how it was read.
bool ReadVsCb(ID3D11DeviceContext* c, UINT slot, UINT bytes, float* out, UINT& off, UINT& size, const char*& how) {
    ID3D11Buffer* b = nullptr;
    UINT first = 0, num = 0;
    if (g_g.c1)
        g_g.c1->VSGetConstantBuffers1(slot, 1, &b, &first, &num);
    else
        c->VSGetConstantBuffers(slot, 1, &b);
    if (!b) return false;
    ComPtr<ID3D11Buffer> buf;
    buf.Attach(b);
    D3D11_BUFFER_DESC bd{};
    b->GetDesc(&bd);
    off = first * 16;
    size = bd.ByteWidth;
    memset(out, 0, bytes);
    bool got = false;
    auto readBack = [&](UINT stSize, bool part) {
        D3D11_BUFFER_DESC sd{};
        sd.ByteWidth = stSize;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> st;
        if (FAILED(g_g.dev->CreateBuffer(&sd, nullptr, &st))) return;
        {
            Internal in;
            if (part) {
                const D3D11_BOX box{off, 0, 0, off + bytes, 1, 1};
                c->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, b, 0, &box);
            } else {
                c->CopyResource(st.Get(), b);
            }
        }
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(c->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return;
        memcpy(out, (const uint8_t*)m.pData + (part ? 0 : off), bytes);
        c->Unmap(st.Get(), 0);
        for (UINT i = 0; i < bytes / 4; ++i) got |= out[i] != 0;
    };
    if (off + bytes <= bd.ByteWidth) {
        readBack((bytes + 255) & ~255u, true);
        how = got ? "part" : "part (all zero)";
        if (!got && bd.ByteWidth <= (256u << 20)) {
            readBack(bd.ByteWidth, false);
            how = got ? "whole buffer" : "whole buffer (all zero)";
        }
    } else {
        how = "out of range";
    }
    return true;
}

// Row-major 4 x 4 inverse (Gauss-Jordan with pivoting); false when singular.
bool Inverse4(const double m[16], double inv[16]) {
    double a[4][8];
    for (int r = 0; r < 4; ++r)
        for (int col = 0; col < 8; ++col) a[r][col] = col < 4 ? m[r * 4 + col] : (col - 4 == r ? 1.0 : 0.0);
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        if (std::fabs(a[piv][col]) < 1e-12) return false;
        if (piv != col)
            for (int k = 0; k < 8; ++k) std::swap(a[piv][k], a[col][k]);
        const double d = a[col][col];
        for (int k = 0; k < 8; ++k) a[col][k] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = a[r][col];
            for (int k = 0; k < 8; ++k) a[r][k] -= f * a[col][k];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int col = 0; col < 4; ++col) inv[r * 4 + col] = a[r][col + 4];
    return true;
}

std::string Mat(const float* m) {
    char b[200];
    snprintf(b, sizeof(b), "[%.6f %.6f %.6f %.6f] [%.6f %.6f %.6f %.6f] [%.6f %.6f %.6f %.6f] [%.6f %.6f %.6f %.6f]", m[0],
             m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
    return b;
}

// COMMON_BUFFER1 (VS slot 13, per draw; capture B2): worldViewProj at byte 0, worldView at 144.
constexpr UINT kObjSlot = 13, kObjBytes = 464, kObjWvp = 0, kObjWv = 144;
constexpr UINT kCamView = 0, kCamPos = 736;

// The camera constants the next draw reads (VS slot 12) and its object constants (slot 13: the projection the vertices
// really get, worldView^-1 * worldViewProj in the row-vector convention), logged once per probed task.
void ProbeCamera(ID3D11DeviceContext* c) {
    float v[kCamBytes / 4];
    UINT off = 0, size = 0;
    const char* how = "none";
    if (!ReadVsCb(c, kCamSlot, kCamBytes, v, off, size, how)) return;  // the next draw then
    g_probe.done = true;
    const float* pm = v + kCamProj / 4;
    const float* ssd = v + kCamSsd / 4;
    const float* vpd = v + kCamVpd / 4;
    const float* cj = v + kCamJitter / 4;
    float fj[2] = {};
    const bool known = FrameJitter(fj);
    const Pass& p = g_p;
    Log("MVR jitter probe (frame %llu): %s %s, VS slot 12 at byte %u of %u (%s): projMatrix %s; cameraJitter (%.6f, %.6f, "
        "%.6f, %.6f); screenSpaceDimensions (%.6f, %.6f, %.8f, %.8f); viewportPixelDimensions (%.1f, %.1f, %.1f, %.1f); "
        "the engine's offset now %s (%.4f, %.4f); MVR pass %s (M jitter %s (%.4f, %.4f), rows x [%.6f %.6f], y "
        "[%.6f %.6f])",
        (unsigned long long)g_frame, kGroupName[g_probe.group], g_probe.name, off, size, how, Mat(pm).c_str(), cj[0],
        cj[1], cj[2], cj[3], ssd[0], ssd[1], ssd[2], ssd[3], vpd[0], vpd[1], vpd[2], vpd[3], known ? "known" : "UNKNOWN",
        fj[0], fj[1], t_pass ? (p.forward ? "forward" : "G-buffer") : "none", p.jitterKnown ? "moved" : "not moved",
        p.jit[0], p.jit[1], p.rows[0][0], p.rows[0][3], p.rows[1][1], p.rows[1][3]);
    const float* view = v + kCamView / 4;
    const float* pos = v + kCamPos / 4;
    float o[kObjBytes / 4];
    UINT ooff = 0, osize = 0;
    const char* ohow = "none";
    if (!ReadVsCb(c, kObjSlot, kObjBytes, o, ooff, osize, ohow)) {
        Log("MVR jitter probe (frame %llu): %s %s: view %s, worldCameraPos (%.6f, %.6f, %.6f); no VS slot 13",
            (unsigned long long)g_frame, kGroupName[g_probe.group], g_probe.name, Mat(view).c_str(), pos[0], pos[1],
            pos[2]);
        return;
    }
    const float* wvp = o + kObjWvp / 4;
    const float* wv = o + kObjWv / 4;
    double dwv[16], inv[16];
    for (int i = 0; i < 16; ++i) dwv[i] = wv[i];
    float peff[16] = {};
    const bool inverted = Inverse4(dwv, inv);
    if (inverted)
        for (int r = 0; r < 4; ++r)
            for (int col = 0; col < 4; ++col) {
                double sum = 0;
                for (int k = 0; k < 4; ++k) sum += inv[r * 4 + k] * wvp[k * 4 + col];
                peff[r * 4 + col] = (float)sum;
            }
    Log("MVR jitter probe (frame %llu): %s %s: view %s; worldCameraPos (%.6f, %.6f, %.6f); first draw's VS slot 13 at byte "
        "%u of %u (%s): worldViewProj %s; worldView %s; their projection worldView^-1 * worldViewProj %s%s",
        (unsigned long long)g_frame, kGroupName[g_probe.group], g_probe.name, Mat(view).c_str(), pos[0], pos[1], pos[2],
        ooff, osize, ohow, Mat(wvp).c_str(), Mat(wv).c_str(), Mat(peff).c_str(), inverted ? "" : " (worldView SINGULAR)");
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

// B15: the corner of slice 1 goes straight into focus's clone of `orig` (no stash copy) when nothing reads that clone
// before the focus group would have got the stash: for eye L always (its clones are focus L's, and outer R's pass writes
// only focus R's twins or its stash); for eye R only into focus R's twin (a shared clone still holds eye L's G-buffer for
// focus L's group). Not in a shot frame or a diagnosis mode, which read the stash.
bool DirectOk(int eye, ID3D11Resource* orig) {
    if (!g_directOn || g_mode != Mode::Skip || g_shot.armed || g_early == 5) return false;
    if (eye == 0) return true;
    return !g_cloneFn && clones::HasTwin(orig);
}

// The corner of slice 1 of copy `cp` straight into focus's clone (true) or nothing written (false).
bool ExtractDirect(ID3D11DeviceContext* c, Copy& cp, ID3D11Texture2D* clone) {
    Pass& p = g_p;
    clones::RawScope raw;  // outer's group: the bind fix would put the original in place of focus's clone
    if (cp.depth) {
        if (!cp.depthSrv) return false;
        DXGI_FORMAT dsvF, srvF;
        DepthFormats(cp.desc.Format, dsvF, srvF);
        if (dsvF == DXGI_FORMAT_UNKNOWN) return false;
        D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = dsvF;
        dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        auto* dsv = static_cast<ID3D11DepthStencilView*>(CloneViewOf(clone, 1, &dv, sizeof(dv)));
        if (!dsv) return false;
        DepthPass(c, cp.depthSrv.Get(), dsv, p.fw, p.fh, DepthBias(), DepthSlope(), DepthDilate());
        return true;
    }
    const D3D11_BOX box{0, 0, 0, p.fw, p.fh, 1};
    Internal in;
    c->CopySubresourceRegion(clone, 0, 0, 0, 0, cp.arr.Get(), D3D11CalcSubresource(0, 1, cp.desc.MipLevels), &box);
    return true;
}

// The corners of slice 1 of the outer-size copies into the eye's stash (or straight into focus's clones, B15).
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
        if (g.tex && !SameLayout(g.desc, cd)) g = StashG{};
        g.orig = cp.orig;
        g.desc = cd;
        g.depth = cp.depth;
        g.direct = false;
        if (DirectOk(p.eye, cp.orig)) {
            g.direct = ExtractDirect(c, cp, clone);
            if (!g.direct) {
                ++g_st.directFails;
                ++g_tot.directFails;
            }
        }
        if (g.direct) {
            ++g_st.direct[p.eye];
            ++g_tot.direct[p.eye];
            ++g_test.direct;
            ++s.n;
            continue;
        }
        if (!g.tex) {
            D3D11_TEXTURE2D_DESC sd = cd;
            sd.MiscFlags &= ~(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                              D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE);
            if (FAILED(g_g.dev->CreateTexture2D(&sd, nullptr, &g.tex))) {
                g = StashG{};
                return PassFailed(p, "stash texture");
            }
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
        if (g.depth) {
            if (!cp.depthSrv) return PassFailed(p, "the depth format has no shader view");
            DepthPass(c, cp.depthSrv.Get(), g.dsv.Get(), p.fw, p.fh, DepthBias(), DepthSlope(), DepthDilate());
        } else {
            const D3D11_BOX box{0, 0, 0, p.fw, p.fh, 1};
            Internal in;
            c->CopySubresourceRegion(g.tex.Get(), 0, 0, 0, 0, cp.arr.Get(),
                                     D3D11CalcSubresource(0, 1, cp.desc.MipLevels), &box);
        }
        ++g_st.viaStash[p.eye];
        ++g_tot.viaStash[p.eye];
        ++s.n;
    }
    if (!s.n) PassFailed(p, "no outer-size target");
}

// The forward pass's corners of slice 1 for the comparison (not into focus's clones): colors by copy, the depth as MVR
// drew it.
void ExtractForward(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    g_fwdN = 0;
    for (Copy& cp : g_copies) {
        if (!cp.used || !cp.filled || cp.desc.Width != p.w || cp.desc.Height != p.h) continue;
        if (g_fwdN >= kMaxTargets) return PassFailed(p, "more than 4 outer-size forward targets");
        if (cp.desc.SampleDesc.Count > 1)
            return PassFailed(p, "a multisampled forward target (the test compares single-sample targets)");
        D3D11_TEXTURE2D_DESC cd = cp.desc;
        cd.Width = p.fw;
        cd.Height = p.fh;
        cd.MipLevels = 1;
        cd.ArraySize = 1;
        cd.MiscFlags = 0;
        cd.BindFlags = cp.depth ? D3D11_BIND_DEPTH_STENCIL : D3D11_BIND_SHADER_RESOURCE;
        StashG& g = g_fwd[g_fwdN];
        if (!g.tex || !SameLayout(g.desc, cd)) {
            g = StashG{};
            if (FAILED(g_g.dev->CreateTexture2D(&cd, nullptr, &g.tex))) return PassFailed(p, "forward stash texture");
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
                    return PassFailed(p, "forward stash depth view");
                }
            }
        }
        g.orig = cp.orig;
        if (g.depth) {
            if (!cp.depthSrv) return PassFailed(p, "the forward depth format has no shader view");
            DepthPass(c, cp.depthSrv.Get(), g.dsv.Get(), p.fw, p.fh, 0, 0.0f, false);
        } else {
            const D3D11_BOX box{0, 0, 0, p.fw, p.fh, 1};
            Internal in;
            c->CopySubresourceRegion(g.tex.Get(), 0, 0, 0, 0, cp.arr.Get(),
                                     D3D11CalcSubresource(0, 1, cp.desc.MipLevels), &box);
        }
        ++g_fwdN;
    }
    if (!g_fwdN) PassFailed(p, "no outer-size forward target");
}

void ExtractForward(ID3D11DeviceContext* c);

// The forward pass's PS slot 3 back to the game's binding, t21..t25 empty.
void RestorePsSlots(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    Internal in;
    clones::RawScope raw;
    c->PSSetConstantBuffers(mvrps::kCbSlot, 1, &p.psSlot3);
    ID3D11ShaderResourceView* none[mvrps::kTextures] = {};
    c->PSSetShaderResources(mvrps::kTexBase, mvrps::kTextures, none);
}

// End of the MVR part: MVR off, slot 3 back, the game's views bound again (rebind; at a tail bind the game binds its
// own next), slice 0 into the game's targets, slice 1's corners into the stash. The rest of the task runs plain.
void EndMvr(ID3D11DeviceContext* c, bool rebind) {
    Pass& p = g_p;
    if (p.forward) FlushPending(c);  // bound but not drawn: slice 0 goes back, the corner out (B12)
    t_pass = false;
    if (p.views != 1) SetViews(c, 1);
    p.views = 1;
    RestoreSlot3(c);
    if (p.forward || p.gbufPsBound) RestorePsSlots(c);
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
    // GPU spans (gpu.h) of the G-buffer pass only; the forward pass's parts are B9d's.
    const bool spans = !p.forward && p.eye >= 0 && p.eye <= 1;
    if (spans) gpu::SpanBegin(p.eye ? gpu::CopyBackR : gpu::CopyBackL);
    {
        Internal in;
        for (Copy& cp : g_copies) {
            if (!cp.used) continue;
            if (cp.native) {  // B17: slice 0 is the game's texture
                ++p.nativeUsed;
                continue;
            }
            for (UINT m = 0; m < cp.desc.MipLevels; ++m)
                c->CopySubresourceRegion(cp.orig, D3D11CalcSubresource(m, 0, cp.desc.MipLevels), 0, 0, 0, cp.arr.Get(),
                                         D3D11CalcSubresource(m, 0, cp.desc.MipLevels), nullptr);
            ++p.copiedBack;
        }
    }
    for (Stats* st : {&g_st, &g_tot}) {
        st->nativeT[p.forward] += p.nativeUsed;
        st->copiedT[p.forward] += p.copiedBack;
    }
    if (spans) gpu::SpanEnd(p.eye ? gpu::CopyBackR : gpu::CopyBackL);
    p.part = PartTail;
    if (p.mapFailures) PassFailed(p, "a target could not be mapped to a copy");
    if (p.viewsFailures) PassFailed(p, "SetMultiViewMode failed");
    if (p.cbGame) PassFailed(p, "the game bound VS slot 3");
    if (!p.failed) {
        if (p.forward) {
            if (!p.live) ExtractForward(c);  // the live pass writes focus's clones at the task's end (LiveEnd)
        } else {
            gpu::SpanBegin(p.eye ? gpu::ExtractR : gpu::ExtractL);
            Extract(c);
            gpu::SpanEnd(p.eye ? gpu::ExtractR : gpu::ExtractL);
        }
    }
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
const char* StartMvr(ID3D11DeviceContext* c, ID3D11Resource* main, int e, uint8_t* task, double t0, bool forward,
                     bool live) {
    Pass& p = g_p;
    const Win& win = g_win[e];
    bool ok = false;
    const D3D11_TEXTURE2D_DESC md = DescOf(main, &ok);
    ID3D11Texture2D* clone = ok ? FocusClone(main) : nullptr;
    D3D11_TEXTURE2D_DESC cd{};
    if (clone) clone->GetDesc(&cd);
    uint32_t sw = 0, sh = 0;  // B12 step 5: a slice clone has focus's size, not its array's
    if (clone && forward && FocusSliceOf(main, e, &sw, &sh)) {
        cd.Width = sw;
        cd.Height = sh;
    }
    // B12: the forward pass may be multisampled (MSAA in the game: its targets go in and out sample by sample); the
    // G-buffer pass's corners are copied as single-sample textures.
    const char* why = !ok ? "no 2D target in outer's pass"
                      : win.w && (md.Width != win.w || md.Height != win.h)
                          ? "the first target of outer's pass is not at the game's size"
                      : md.ArraySize != 1 && !((md.ArraySize == 2 || (forward && md.ArraySize == 3)) && IsNative(main))
                          ? "outer's depth is an array"
                      : md.SampleDesc.Count != 1 && !forward                       ? "outer's depth is multisampled"
                      : !clone                                         ? "no focus clone of outer's depth"
                      : cd.SampleDesc.Count != md.SampleDesc.Count || cd.SampleDesc.Quality != md.SampleDesc.Quality
                          ? "focus's depth clone has another sample count"
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
    p.forward = forward;
    p.live = live;
    if (live) g_otherClears.clear();
    for (Copy& cp : g_copies) cp.used = cp.filled = cp.pending = false;
    {
        Internal in;
        if (g_g.c1)
            g_g.c1->VSGetConstantBuffers1(mvrvs::kCbSlot, 1, &p.slot3, &p.slot3First, &p.slot3Num);
        else
            c->VSGetConstantBuffers(mvrvs::kCbSlot, 1, &p.slot3);
        ID3D11Buffer* cb = g_g.rowsCb.Get();
        c->VSSetConstantBuffers(mvrvs::kCbSlot, 1, &cb);
        if (forward) {
            c->PSGetConstantBuffers(mvrps::kCbSlot, 1, &p.psSlot3);
            ID3D11Buffer* pcb = g_g.psRowsCb.Get();
            c->PSSetConstantBuffers(mvrps::kCbSlot, 1, &pcb);
            g_srvClones.clear();
        }
    }
    ID3D11VertexShader* vs = nullptr;
    c->VSGetShader(&vs, nullptr, nullptr);
    const bool patched = mvrvs::IsPatched(vs);
    if (vs) vs->Release();
    p.vsPatched = patched;
    FillM(c);
    // The forward pass picks the views at each draw (its PS counts too).
    if (!forward && patched && SetViews(c, 2)) p.views = 2;
    t_pass = true;
    return nullptr;
}

// A pass that runs plain: the G-buffer's focus group runs its own pass, the forward test has no comparison, the live
// forward's focus group runs its own forward pass.
void PassRefused(bool forward, bool live, int e, const char* why) {
    if (live)
        FwdReason(e, why);
    else if (forward)
        FwdRefused(why);
    else
        Own(e, why);
}

// Outer's G-buffer pass (or, forward = true, outer's forward pass: outer L's of the forward test, B9c, or either eye's
// live one, B9d) begins.
void PassBegin(uint8_t* task, int e, bool forward, bool live = false) {
    Pass& p = g_p;
    if (!forward) g_stash[e].valid = false;
    if (p.part != PartNone) {  // a pass whose task did not end (should not happen)
        ReleasePassRefs(p);
        p = Pass{};
    }
    if (!g_enabled) return;
    const Win& win = g_win[e];
    if (!win.valid || win.frame != g_frame) return PassRefused(forward, live, e, "no focus window in outer's pass");
    ID3D11DeviceContext* c = Imm();
    if (!c || !EnsureGpu(c)) return PassRefused(forward, live, e, "no D3D11 objects");
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
        p.forward = forward;
        p.live = live;
        return;
    }
    if (const char* why = StartMvr(c, main.Get(), e, task, NowMs(), forward, live)) {
        release();
        return PassRefused(forward, live, e, why);
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
    const bool forward = p.forward, live = p.live;
    if (const char* why = StartMvr(c, main, e, p.task, p.t0, forward, live)) {
        p = Pass{};
        PassRefused(forward, live, e, why);
        return false;
    }
    g_p.lateStarts = 1;
    g_p.waitBinds = waited;
    return true;
}

void ShotOuter(ID3D11DeviceContext* c);
void ForwardEnd(ID3D11DeviceContext* c);
void ShotFocusBefore(ID3D11DeviceContext* c);
void ShotFocusAfter(ID3D11DeviceContext* c);
void ReportShot();

void PassEnd(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    if (p.forward) return ForwardEnd(c);
    if (p.part == PartWait) {  // the task bound nothing
        const int e = p.eye;
        p = Pass{};
        return Own(e, "outer's pass bound no target at the game's size");
    }
    if (p.part == PartMvr) EndMvr(c, true);
    Stash& s = g_stash[p.eye];
    s.nm = 0;
    const gpu::Span maskSpan = p.eye ? gpu::MaskStashR : gpu::MaskStashL;
    gpu::SpanBegin(maskSpan);
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
    gpu::SpanEnd(maskSpan);
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
        st->gbufPsDraws += p.gbufPsDraws;
        st->gbufPsKept += p.gbufPsKept;
    }
    TestState& t = g_test;
    ++t.passes;
    t.gbufPsDraws = p.gbufPsDraws;
    t.gbufPsKept = p.gbufPsKept;
    t.liveTrack = p.liveTrackOk;
    t.srvMisses = p.srvMisses;
    t.binds = p.binds;
    t.clears = p.clears;
    t.mapFailures = p.mapFailures;
    t.patchedSets = p.patchedSets;
    t.plainSets = p.plainSets;
    t.cbRebinds = p.cbRebinds;
    t.copiedBack = p.copiedBack;
    t.native = p.nativeUsed;
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
        g_shot.pass.psSlot3 = nullptr;
        g_shot.pass.masks.clear();
        ShotOuter(c);
    }
    ReleasePassRefs(p);
    p = Pass{};
}

// Mode 3: the conservative copy of the eye's forward depth for the focus group's velocity pass (VelocityDsv). On failure
// the velocity pass tests against the exact depth (no copy this frame).
void VelocityDepth(ID3D11DeviceContext* c, Copy* cp, ID3D11Texture2D* clone) {
    const int e = g_p.eye;
    VelDepth& v = g_vel[e];
    v.frame = 0;
    D3D11_TEXTURE2D_DESC d{}, have{};
    clone->GetDesc(&d);
    d.MiscFlags &= ~(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                     D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_GDI_COMPATIBLE);
    if (v.tex) v.tex->GetDesc(&have);
    if (!v.tex || !SameLayout(have, d)) {
        v.tex.Reset();
        if (FAILED(g_g.dev->CreateTexture2D(&d, nullptr, &v.tex))) return;
    }
    DXGI_FORMAT dsvF, srvF;
    DepthFormats(d.Format, dsvF, srvF);
    D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
    dv.Format = dsvF;
    dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    auto* dsv = dsvF == DXGI_FORMAT_UNKNOWN ? nullptr
                                            : (ID3D11DepthStencilView*)CloneViewOf(v.tex.Get(), 1, &dv, sizeof(dv));
    if (!dsv) return;
    DepthPass(c, cp->depthSrv.Get(), dsv, g_p.fw, g_p.fh, DepthBias(), DepthSlope(), DepthDilate());
    v.orig = cp->orig;
    v.frame = g_frame;
}

bool IsRg16(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_R16G16_TYPELESS || f == DXGI_FORMAT_R16G16_FLOAT || f == DXGI_FORMAT_R16G16_UNORM ||
           f == DXGI_FORMAT_R16G16_SNORM || f == DXGI_FORMAT_R16G16_UINT || f == DXGI_FORMAT_R16G16_SINT;
}

// Mode 3, a bind in a focus group: the velocity pass (an R16G16 target of the depth's size with focus's depth clone)
// gets the conservative copy of this frame in place of the depth clone, with the game view's description. Null: as is.
ID3D11DepthStencilView* VelocityDsv(ID3D11RenderTargetView* rt, ID3D11DepthStencilView* ds) {
    if (g_curFocus < 0 || g_curFocus > 1) return nullptr;
    VelDepth& v = g_vel[g_curFocus];
    if (v.frame != g_frame || !v.tex || !FocusSwapped()) return nullptr;
    ComPtr<ID3D11Resource> dr, rr;
    ds->GetResource(&dr);
    if (!dr || dr.Get() != (ID3D11Resource*)FocusClone(v.orig)) return nullptr;
    rt->GetResource(&rr);
    bool ok = false;
    const D3D11_TEXTURE2D_DESC rd = DescOf(rr.Get(), &ok);
    D3D11_TEXTURE2D_DESC vd{};
    v.tex->GetDesc(&vd);
    if (!ok || !IsRg16(rd.Format) || rd.Width != vd.Width || rd.Height != vd.Height) return nullptr;
    D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
    ds->GetDesc(&dd);
    return (ID3D11DepthStencilView*)CloneViewOf(v.tex.Get(), 1, &dd, sizeof(dd));
}

// B9d: the corners of slice 1 of the live forward pass into focus's clones (colors by copy, the depth by
// mvr_forward_depth: as drawn by default; a farther depth thins out the crowd in the stands, run 3), then its clears of
// other-size targets on their clones. Everything is checked before the first
// write, so a refused pass leaves the clones as the early run left them.
void ForwardToClones(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    struct Target {
        Copy* cp;
        ID3D11Texture2D* clone;
        ID3D11DepthStencilView* dsv;
        ID3D11RenderTargetView* rtv;  // B12: a multisampled color (the clone's view; the corner by PSColorOutMs)
        bool ms;
    };
    std::vector<Target> ts;
    std::vector<Copy*> sliced;  // B12 step 5: focus's clone is the slice view 1 drew into (nothing to copy)
    for (Copy& cp : g_copies) {
        if (!cp.used || !cp.filled || cp.desc.Width != p.w || cp.desc.Height != p.h) continue;
        if (const int fs = FocusSliceOfCopy(cp)) {
            if (fs != (int)SliceFor(cp))
                return PassFailed(p, "focus's clone of a forward target is another slice than view 1 drew into");
            sliced.push_back(&cp);
            continue;
        }
        if (SliceFor(cp) != 1) return PassFailed(p, "view 1 in slice 2 of a forward target without a slice clone");
        ID3D11Texture2D* clone = FocusClone(cp.orig);
        if (!clone) return PassFailed(p, "a forward target without a focus clone");
        D3D11_TEXTURE2D_DESC cd{};
        clone->GetDesc(&cd);
        if (cd.Width != p.fw || cd.Height != p.fh || cd.Format != cp.desc.Format ||
            cd.SampleDesc.Count != cp.desc.SampleDesc.Count || cd.SampleDesc.Quality != cp.desc.SampleDesc.Quality ||
            cd.ArraySize != 1)
            return PassFailed(p, "a forward target's focus clone of another size or format");
        const bool ms = cd.SampleDesc.Count > 1;
        ID3D11DepthStencilView* dsv = nullptr;
        ID3D11RenderTargetView* rtv = nullptr;
        if (cp.depth) {
            if (!cp.depthSrv) return PassFailed(p, "the forward depth format has no shader view");
            DXGI_FORMAT dsvF, srvF;
            DepthFormats(cd.Format, dsvF, srvF);
            D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
            dv.Format = dsvF;
            dv.ViewDimension = ms ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;
            dsv = dsvF == DXGI_FORMAT_UNKNOWN ? nullptr
                                              : (ID3D11DepthStencilView*)CloneViewOf(clone, 1, &dv, sizeof(dv));
            if (!dsv) return PassFailed(p, "no depth view on focus's depth clone");
        } else if (ms) {
            if (cp.viewFmt == DXGI_FORMAT_UNKNOWN || !(cp.desc.BindFlags & D3D11_BIND_SHADER_RESOURCE))
                return PassFailed(p, "a multisampled forward color without a shader view");
            if (!cp.colorSrv) {
                D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
                sv.Format = cp.viewFmt;
                sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
                sv.Texture2DMSArray = {0, 2};
                if (FAILED(g_g.dev->CreateShaderResourceView(cp.arr.Get(), &sv, &cp.colorSrv)))
                    return PassFailed(p, "no shader view on a multisampled forward color");
            }
            D3D11_RENDER_TARGET_VIEW_DESC rv{};
            rv.Format = cp.viewFmt;
            rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
            rtv = (ID3D11RenderTargetView*)CloneViewOf(clone, 0, &rv, sizeof(rv));
            if (!rtv) return PassFailed(p, "no target view on a multisampled color clone");
        }
        ts.push_back({&cp, clone, dsv, rtv, ms});
    }
    if (ts.empty() && sliced.empty()) return PassFailed(p, "no outer-size forward target");
    struct Clear {
        const OtherClear* oc;
        ID3D11View* v;
    };
    std::vector<Clear> cs;
    unsigned misses = 0;
    for (const OtherClear& oc : g_otherClears) {
        ID3D11Texture2D* clone = FocusClone(oc.orig);
        if (!clone) {  // a texture focus shares with outer: the game's clear went to it already
            ++misses;
            continue;
        }
        ID3D11View* v = oc.depth ? CloneViewOf(clone, 1, &oc.dd, sizeof(oc.dd)) : CloneViewOf(clone, 0, &oc.rd, sizeof(oc.rd));
        if (!v) return PassFailed(p, "no view on the focus clone of a cleared target of another size");
        cs.push_back({&oc, v});
    }
    clones::RawScope raw;  // outer's group: the bind fix would put the originals in place of focus's clones
    const gpu::Span span = p.eye ? gpu::FwdOutR : gpu::FwdOutL;
    gpu::SpanBegin(span);
    for (const Target& t : ts) {
        if (t.dsv) {
            const bool stash = g_fwdDepth == 2, bias = g_fwdDepth == 1 || stash;
            DepthPass(c, t.cp->depthSrv.Get(), t.dsv, p.fw, p.fh, bias ? DepthBias() : 0u,
                      stash ? DepthSlope() : 0.0f, stash && DepthDilate(), t.ms ? g_g.psDepthMs.Get() : nullptr);
            // Mode 3's copy for the velocity pass: a single-sample R16G16 target cannot take a multisampled depth.
            if (g_fwdDepth == 3 && !t.ms) VelocityDepth(c, t.cp, t.clone);
        } else if (t.ms) {
            ColorPassMs(c, t.cp->colorSrv.Get(), t.rtv, p.fw, p.fh, g_g.psColorOutMs.Get());
        } else {
            const D3D11_BOX box{0, 0, 0, p.fw, p.fh, 1};
            Internal in;
            c->CopySubresourceRegion(t.clone, 0, 0, 0, 0, t.cp->arr.Get(), D3D11CalcSubresource(0, 1, t.cp->desc.MipLevels),
                                     &box);
        }
        NoteOrig(t.cp->orig);
    }
    for (Copy* cp : sliced) NoteOrig(cp->orig);  // focus R's slice of it (the twin) is wanted the same way
    gpu::SpanEnd(span);
    {
        Internal in;
        for (const Clear& k : cs) {
            if (k.oc->depth)
                c->ClearDepthStencilView((ID3D11DepthStencilView*)k.v, k.oc->flags, k.oc->d, k.oc->s);
            else
                c->ClearRenderTargetView((ID3D11RenderTargetView*)k.v, k.oc->col);
            NoteOrig(k.oc->orig);
        }
    }
    for (FwdStats* st : {&g_fs, &g_ftot}) {
        st->otherClears += cs.size();
        st->otherClearMisses += misses;
    }
    g_testLive.otherClears += (unsigned)cs.size();
    g_testLive.otherClearMisses += misses;
}

// End of a live forward pass (B9d): view 1 into focus's clones, or the focus group runs its own pass.
void LiveEnd(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    const int e = p.eye;
    if (p.part == PartWait) {
        p = Pass{};
        return FwdReason(e, "outer's forward pass bound no target at the game's size");
    }
    if (p.part == PartMvr) EndMvr(c, true);
    if (p.lateDraws) PassFailed(p, "draws into an outer-size target after the MVR part");
    if (p.pausedDraws) PassFailed(p, "draws into targets of other sizes (focus would miss them)");
    if (p.plainDraws > p.pausedDraws) PassFailed(p, "draws with one view (VS or PS without the patch)");
    if (!p.failed) ForwardToClones(c);
    const double ms = NowMs() - p.t0;
    for (FwdStats* st : {&g_fs, &g_ftot}) {
        ++st->passes[e];
        st->ms[e] += ms;
        st->draws += p.draws;
        st->copyIns += p.copyIns;
        st->clearFills += p.clearFills;
        st->msTargets += p.msTargets;
        st->colorSkips += p.colorSkips;
        st->sliceTargets[e] += p.sliceTargets;
        st->sliceClears += p.sliceClears;
        if (!p.failed) ++st->ok[e];
    }
    ++g_testLive.passes;
    g_testLive.native = p.nativeUsed;
    g_testLive.copiedBack = p.copiedBack;
    g_testLive.copyIns = p.copyIns;
    g_testLive.clearFills = p.clearFills;
    g_testLive.msTargets = p.msTargets;
    g_testLive.colorSkips = p.colorSkips;
    g_testLive.sliceTargets = p.sliceTargets;
    g_testLive.sliceClears = p.sliceClears;
    if (!p.failed) {
        g_fwdOk[e] = g_frame;
        ++g_testLive.ok;
    } else {
        g_testLive.why = p.why;
        FwdReason(e, p.why.c_str());
        if (p.why != g_lastFwdFail) {
            g_lastFwdFail = p.why;
            Log("MVR forward: outer %s's pass left focus's forward to the focus group in frame %llu: %s (logged once "
                "per reason; draws %u, one view %u, paused %u times with %u draws)",
                kEye[e], (unsigned long long)g_frame, p.why.c_str(), p.draws, p.plainDraws, p.pauses, p.pausedDraws);
        }
    }
    ReleasePassRefs(p);
    p = Pass{};
}

// End of outer L's forward pass (B9c): the corners are kept for the comparison after focus L's own forward pass.
void ForwardEnd(ID3D11DeviceContext* c) {
    Pass& p = g_p;
    if (p.live) return LiveEnd(c);
    FwdShot& sh = g_fshot;
    if (p.part == PartWait) {
        p = Pass{};
        return FwdRefused("outer L's forward pass bound no target at the game's size");
    }
    if (p.part == PartMvr) EndMvr(c, true);
    if (p.lateDraws) PassFailed(p, "draws into an outer-size target after the MVR part");
    sh.outerDone = true;
    sh.ms = NowMs() - p.t0;
    sh.pass = p;
    sh.pass.slot3 = nullptr;
    sh.pass.psSlot3 = nullptr;
    sh.pass.masks.clear();
    sh.n = p.failed ? 0 : g_fwdN;
    if (sh.base >= 0)
        for (int k = 0; k < sh.n; ++k) shots::Grab(c, g_fwd[k].tex.Get(), sh.base + 2 * k + 1);
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
    const gpu::Span span = eye ? gpu::FillR : gpu::FillL;
    gpu::SpanBegin(span);
    Internal in;
    for (int k = 0; k < s.n; ++k) {
        if (s.g[k].depth && !depth) continue;
        if (ID3D11Texture2D* t = FocusClone(s.g[k].orig)) {
            if (!s.g[k].direct) c->CopyResource(t, s.g[k].tex.Get());  // B15: else the pass wrote it already
            NoteOrig(s.g[k].orig);
        } else {
            failed = true;
        }
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
        if (t != hold.Get()) NoteOrig(m.orig);  // a clone (else the game's own texture)
        ++g_st.masks;
        ++g_tot.masks;
    }
    gpu::SpanEnd(span);
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
// `what`: the line's start ("MVR G-buffer shot: target 3"), `fix`: how ours was made, `pass`: its jitter.
void CompareDepth(ID3D11DeviceContext* c, const char* what, const char* fix, ID3D11Texture2D* own, ID3D11Texture2D* ours,
                  const Pass& pass) {
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
    Log("%s (depth, %s) focus L own vs ours, bit for bit: same %.3f%%, ours "
        "farther %.3f%% (max %lld steps), nearer %.3f%% (max %lld); |steps| 1: %.3f%%, 2-4: %.3f%%, 5-16: %.3f%%, 17-64: "
        "%.3f%%, 65-1024: %.3f%%, more: %.3f%%; far plane in focus's own %.2f%%",
        what, fix, same / px, farther / px, (long long)maxFar, nearer / px, (long long)maxNear, bucket[0] / px,
        bucket[1] / px, bucket[2] / px, bucket[3] / px, bucket[4] / px, bucket[5] / px, far1 / px);
    Log("%s: ours - own fitted as own's slope times a shift over %llu smooth pixels: shift "
        "(%.4f, %.4f) px (ours = own sampled there), rms %.3g, left after the shift %.3g; the pass's jitter offset "
        "(%.4f, %.4f), M %s",
        what, (unsigned long long)fitN, shx, shy, rms, resid, pass.jit[0], pass.jit[1],
        pass.jitterKnown ? "moved to focus's own" : "with outer's jitter");
}

// A color texel as linear floats (R11G11B10_FLOAT, R16G16B16A16_FLOAT, R8G8B8A8); false: format not read.
float HalfToFloat(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const float sign = (h & 0x8000) ? -1.0f : 1.0f;
    if (e == 0) return sign * std::ldexp((float)m, -24);
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * std::ldexp((float)(m | 0x400), (int)e - 25);
}
float SmallFloat(uint32_t v, int mbits) {  // unsigned 5-bit exponent, mbits mantissa (R11G11B10)
    const uint32_t m = v & ((1u << mbits) - 1), e = v >> mbits;
    if (e == 0) return std::ldexp((float)m, -14 - mbits);
    if (e == 31) return m ? NAN : INFINITY;
    return std::ldexp((float)(m | (1u << mbits)), (int)e - 15 - mbits);
}
bool Texel(DXGI_FORMAT f, const uint8_t* row, UINT x, float out[3]) {
    switch (f) {
    case DXGI_FORMAT_R11G11B10_FLOAT: {
        uint32_t v;
        memcpy(&v, row + 4 * x, 4);
        out[0] = SmallFloat(v & 0x7FF, 6);
        out[1] = SmallFloat((v >> 11) & 0x7FF, 6);
        out[2] = SmallFloat(v >> 22, 5);
        return true;
    }
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT: {
        uint16_t h[4];
        memcpy(h, row + 8 * x, 8);
        for (int i = 0; i < 3; ++i) out[i] = HalfToFloat(h[i]);
        return true;
    }
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        for (int i = 0; i < 3; ++i) out[i] = row[4 * x + i] / 255.0f;
        return true;
    default: return false;
    }
}

// A shot's color target: focus's own vs ours, relative to the brighter of the two (plus 2% of own's mean luminance, so
// dark pixels do not count small absolute steps), and where the differences lie (64 x 64 blocks).
void CompareColor(ID3D11DeviceContext* c, const char* what, ID3D11Texture2D* own, ID3D11Texture2D* ours) {
    const ComPtr<ID3D11Texture2D> a = StagingOf(c, own), b = StagingOf(c, ours);
    if (!a || !b) return;
    D3D11_TEXTURE2D_DESC d{};
    a->GetDesc(&d);
    float probe[3];
    const uint8_t zero[16] = {};
    if (!Texel(d.Format, zero, 0, probe)) {
        Log("%s: format %d not compared (the PNG pair still is)", what, (int)d.Format);
        return;
    }
    D3D11_MAPPED_SUBRESOURCE ma{}, mb{};
    if (FAILED(c->Map(a.Get(), 0, D3D11_MAP_READ, 0, &ma))) return;
    if (FAILED(c->Map(b.Get(), 0, D3D11_MAP_READ, 0, &mb))) {
        c->Unmap(a.Get(), 0);
        return;
    }
    auto luma = [](const float v[3]) { return 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2]; };
    double sumOwn = 0;
    uint64_t finite = 0;
    for (UINT y = 0; y < d.Height; ++y) {
        const uint8_t* ra = (const uint8_t*)ma.pData + (size_t)y * ma.RowPitch;
        for (UINT x = 0; x < d.Width; ++x) {
            float va[3];
            Texel(d.Format, ra, x, va);
            const float l = luma(va);
            if (std::isfinite(l)) {
                sumOwn += l;
                ++finite;
            }
        }
    }
    const double meanOwn = finite ? sumOwn / finite : 0;
    const float floorV = (float)(0.02 * meanOwn) + 1e-6f;
    constexpr UINT kBlock = 64;
    const UINT bw = (d.Width + kBlock - 1) / kBlock, bh = (d.Height + kBlock - 1) / kBlock;
    std::vector<uint32_t> blocks((size_t)bw * bh);
    uint64_t same = 0, over[3] = {}, nanOwn = 0, nanOurs = 0;
    double sumAbs = 0, sumSigned = 0;
    for (UINT y = 0; y < d.Height; ++y) {
        const uint8_t* ra = (const uint8_t*)ma.pData + (size_t)y * ma.RowPitch;
        const uint8_t* rb = (const uint8_t*)mb.pData + (size_t)y * mb.RowPitch;
        const size_t texel = d.Format == DXGI_FORMAT_R11G11B10_FLOAT || d.Format >= DXGI_FORMAT_R8G8B8A8_TYPELESS ? 4 : 8;
        for (UINT x = 0; x < d.Width; ++x) {
            if (!memcmp(ra + texel * x, rb + texel * x, texel)) {
                ++same;
                continue;
            }
            float va[3], vb[3];
            Texel(d.Format, ra, x, va);
            Texel(d.Format, rb, x, vb);
            const float la = luma(va), lb = luma(vb);
            if (!std::isfinite(la)) ++nanOwn;
            if (!std::isfinite(lb)) ++nanOurs;
            if (!std::isfinite(la) || !std::isfinite(lb)) continue;
            float rel = 0;
            for (int i = 0; i < 3; ++i)
                rel = std::max(rel, std::fabs(va[i] - vb[i]) / (std::max(std::fabs(va[i]), std::fabs(vb[i])) + floorV));
            if (rel > 0.01f) ++over[0];
            if (rel > 0.05f) {
                ++over[1];
                ++blocks[(size_t)(y / kBlock) * bw + x / kBlock];
            }
            if (rel > 0.25f) ++over[2];
            sumAbs += std::fabs(lb - la);
            sumSigned += lb - la;
        }
    }
    c->Unmap(a.Get(), 0);
    c->Unmap(b.Get(), 0);
    std::vector<size_t> idx(blocks.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return blocks[x] > blocks[y]; });
    std::string worst;
    for (size_t i = 0; i < idx.size() && i < 6 && blocks[idx[i]]; ++i) {
        char t[64];
        snprintf(t, sizeof(t), "%s(%u, %u) %.1f%%", worst.empty() ? "" : ", ", (unsigned)(idx[i] % bw) * kBlock,
                 (unsigned)(idx[i] / bw) * kBlock, blocks[idx[i]] * 100.0 / (kBlock * kBlock));
        worst += t;
    }
    const double px = (double)d.Width * d.Height / 100.0, n = (double)d.Width * d.Height;
    Log("%s (%ux%u fmt %d) focus L own vs ours: bit for bit same %.3f%%; relative difference > 1%% %.3f%%, > 5%% %.3f%%, "
        "> 25%% %.3f%%; luminance: own's mean %.4g, mean |ours - own| %.4g (%.2f%% of the mean), mean ours - own %.4g; "
        "NaN or infinity own %llu, ours %llu; 64 x 64 blocks with the most pixels > 5%% (x, y of the corner): %s",
        what, d.Width, d.Height, (int)d.Format, same / px, over[0] / px, over[1] / px, over[2] / px, meanOwn, sumAbs / n,
        meanOwn > 0 ? 100.0 * sumAbs / n / meanOwn : 0.0, sumSigned / n, (unsigned long long)nanOwn,
        (unsigned long long)nanOurs, worst.empty() ? "none" : worst.c_str());
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
        if (t && s.g[k].tex && s.g[k].depth && (f == DXGI_FORMAT_R32_TYPELESS || f == DXGI_FORMAT_D32_FLOAT)) {
            char what[48], fix[96];
            snprintf(what, sizeof(what), "MVR G-buffer shot: target %d", k);
            float slope = 0;
            memcpy(&slope, &g_g.depthCbData[1], sizeof(slope));
            snprintf(fix, sizeof(fix), "bias %u steps + %.3f px of slope%s", g_g.depthCbData[0], slope,
                     g_g.depthCbData[4] ? ", 3 x 3 farthest" : "");
            CompareDepth(c, what, fix, t, s.g[k].tex.Get(), g_shot.pass);
        }
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

// After focus L's own CPhase3Render in a forward test shot frame: its targets vs the corners of outer L's view 1.
void ReportForward();
void ShotForwardFocus(ID3D11DeviceContext* c) {
    FwdShot& sh = g_fshot;
    sh.focusDone = true;
    for (int k = 0; k < sh.n; ++k) {
        const StashG& g = g_fwd[k];
        ID3D11Texture2D* t = FocusSliceOf(g.orig, 0) ? nullptr : FocusClone(g.orig);  // B12 step 5: no texture
        const D3D11_TEXTURE2D_DESC& d = g.desc;
        snprintf(g_fwdLabels[k], sizeof(g_fwdLabels[k]), "MVR forward %d (%ux%u fmt %d%s): focus L vs view 1 of outer L", k,
                 d.Width, d.Height, (int)d.Format, g.depth ? ", depth" : "");
        if (!t) {
            Log("MVR forward test shot: target %d lost its focus clone", k);
            continue;
        }
        char what[48];
        snprintf(what, sizeof(what), "MVR forward test shot: target %d", k);
        const DXGI_FORMAT f = d.Format;
        if (g.depth && (f == DXGI_FORMAT_R32_TYPELESS || f == DXGI_FORMAT_D32_FLOAT))
            CompareDepth(c, what, "as MVR drew it", t, g.tex.Get(), sh.pass);
        else if (!g.depth)
            CompareColor(c, what, t, g.tex.Get());
        if (sh.base >= 0) shots::Grab(c, t, sh.base + 2 * k);
    }
    ReportForward();
}

void ReportForward() {
    FwdShot& sh = g_fshot;
    if (!sh.armed || sh.reported) return;
    sh.reported = true;
    if (!sh.outerDone) {
        Log("MVR forward test shot (frame %llu): no comparison: %s", (unsigned long long)sh.frame,
            sh.why.empty() ? "outer L's CPhase3Render did not run in the shot frame" : sh.why.c_str());
        return;
    }
    const Pass& p = sh.pass;
    const std::string state = p.failed ? "FAILED: " + p.why : "ok";
    Log("MVR forward test shot (frame %llu): outer L's CPhase3Render with MVR %s, %.3f ms; %d outer-size target(s) "
        "stashed; binds %u (tail %u, draws there into outer-size targets %u), clears %u, copy-ins %u, map failures %u; "
        "draws %u, with one view %u (VS unpatched %u, PS refused %u, PS not made through the patch %u); patched PS "
        "draws %u (PS slot 3 rebinds %u, screen textures without a focus clone %u); VS sets patched %u, unpatched %u; "
        "VS slot 3 rebinds %u (game buffers %u); copied back %u; SetMultiViewMode failures %u; focus L's own forward pass "
        "for the comparison: %s; paused at binds of other sizes %u times (draws there %u, clears %u)",
        (unsigned long long)sh.frame, state.c_str(), sh.ms, sh.n, p.binds, p.tailBinds, p.lateDraws, p.clears,
        p.copyIns, p.mapFailures, p.draws, p.plainDraws, p.vsPlainDraws, p.psOriginalDraws, p.psUnknownDraws,
        p.psPatchedDraws, p.psCbRebinds, p.srvMisses, p.patchedSets, p.plainSets, p.cbRebinds, p.cbGame, p.copiedBack,
        p.viewsFailures, sh.focusDone ? "yes" : "NO", p.pauses, p.pausedDraws, p.pausedClears);
    float r[4][4];
    PsRows(p, r);
    const char* jit = p.jitterKnown ? "moved to focus's own" : Cfg().mvrJitterFix ? "UNKNOWN (outer's kept)"
                                                                                  : "outer's (mvr_jitter_fix=0)";
    Log("MVR forward test shot: view 1 = focus L in the corner %u x %u of %u x %u; jitter %s (offset %.4f, %.4f); M rows "
        "[%.6f 0 0 %.6f] [0 %.6f 0 %.6f]; PS rows: screen inputs S (%.6f, %.6f) O (%.6f, %.6f) with the game's screen "
        "UV scale taken as 1, UV scale (%.6f, %.6f)",
        p.fw, p.fh, p.w, p.h, jit, p.jit[0], p.jit[1], p.rows[0][0], p.rows[0][3], p.rows[1][1], p.rows[1][3], r[0][0],
        r[0][1], r[0][2], r[0][3], r[1][0], r[1][1]);
}

// ---------------------------------------------------------------- task map (shot frame)

struct TaskMap {
    bool on = false, detail = false;
    char group[16] = "", name[48] = "";
    unsigned draws = 0, binds = 0, clears = 0;
    ID3D11Resource* seen[10] = {};  // only compared
    int nseen = 0;
    bool more = false;
    std::string targets;
    // detail: distinct (slot, resource) of the PS shader resources and constant buffers at the draws
    std::vector<std::pair<UINT, void*>> srvs, cbs;
    bool srvMore = false;
};
TaskMap g_map;

// A short id of a resource for comparing map lines (low bits of its address), with "c" when it is a clone.
std::string ResId(void* r) {
    char b[24];
    snprintf(b, sizeof(b), "@%05llx%s", (unsigned long long)(((uintptr_t)r >> 4) & 0xfffff),
             clones::Original(r) != r ? "c" : "");
    return b;
}

void MapDetail(ID3D11DeviceContext* c) {
    constexpr UINT kSrvs = 32, kCbs = 14;
    ID3D11ShaderResourceView* srv[kSrvs] = {};
    c->PSGetShaderResources(0, kSrvs, srv);
    for (UINT i = 0; i < kSrvs; ++i) {
        if (!srv[i]) continue;
        ComPtr<ID3D11Resource> r;
        srv[i]->GetResource(&r);
        srv[i]->Release();
        const std::pair<UINT, void*> k{i, r.Get()};
        if (std::find(g_map.srvs.begin(), g_map.srvs.end(), k) != g_map.srvs.end()) continue;
        if (g_map.srvs.size() >= 96) {
            g_map.srvMore = true;
            continue;
        }
        g_map.srvs.push_back(k);
    }
    ID3D11Buffer* cb[kCbs] = {};
    c->PSGetConstantBuffers(0, kCbs, cb);
    for (UINT i = 0; i < kCbs; ++i) {
        if (!cb[i]) continue;
        const std::pair<UINT, void*> k{i, cb[i]};
        cb[i]->Release();
        if (std::find(g_map.cbs.begin(), g_map.cbs.end(), k) == g_map.cbs.end() && g_map.cbs.size() < 48)
            g_map.cbs.push_back(k);
    }
}

void MapNote(ID3D11View* v, const char* kind) {
    if (!v) return;
    ComPtr<ID3D11Resource> r;
    v->GetResource(&r);
    for (int i = 0; i < g_map.nseen; ++i)
        if (g_map.seen[i] == r.Get()) return;
    if (g_map.nseen == (int)std::size(g_map.seen)) {
        g_map.more = true;
        return;
    }
    g_map.seen[g_map.nseen++] = r.Get();
    bool ok = false;
    const D3D11_TEXTURE2D_DESC d = DescOf(r.Get(), &ok);
    char b[96];
    if (ok)
        snprintf(b, sizeof(b), "%s%s %ux%u f%d x%u%s%s", g_map.targets.empty() ? "" : ", ", kind, d.Width, d.Height,
                 (int)d.Format, d.SampleDesc.Count, d.ArraySize > 1 ? " array" : "",
                 g_map.detail ? (" " + ResId(r.Get())).c_str() : "");
    else
        snprintf(b, sizeof(b), "%s%s (not 2D)", g_map.targets.empty() ? "" : ", ", kind);
    g_map.targets += b;
}

void MapBind(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv) {
    if (!g_map.on || t_internal || c != Imm()) return;
    ++g_map.binds;
    for (UINT i = 0; i < n && rtv; ++i) MapNote(rtv[i], "rt");
    MapNote(dsv, "ds");
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
        if (p.forward) {  // views per draw (ForwardDraw)
            p.vsPatched = mvrvs::IsPatched(vs);
            if (p.vsPatched) {
                ++p.patchedSets;
                FillM(c);
            } else {
                ++p.plainSets;
            }
            return;
        }
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
    if (SkipDraw(c)) return;
    BeforeDraw(c);
    o_dii(c, a, b, d, e, f);
}
void STDMETHODCALLTYPE D_DrawInstanced(ID3D11DeviceContext* c, UINT a, UINT b, UINT d, UINT e) {
    if (SkipDraw(c)) return;
    BeforeDraw(c);
    o_di(c, a, b, d, e);
}
void STDMETHODCALLTYPE D_DrawAuto(ID3D11DeviceContext* c) {
    if (SkipDraw(c)) return;
    BeforeDraw(c);
    o_da(c);
}
void STDMETHODCALLTYPE D_DrawIndexedInstancedIndirect(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT o) {
    if (SkipDraw(c)) return;
    BeforeDraw(c);
    o_diii(c, b, o);
}
void STDMETHODCALLTYPE D_DrawInstancedIndirect(ID3D11DeviceContext* c, ID3D11Buffer* b, UINT o) {
    if (SkipDraw(c)) return;
    BeforeDraw(c);
    o_dii2(c, b, o);
}

bool CheckVtable(uintptr_t base, uintptr_t vt, uintptr_t exec, const char* name, bool fatal = true) {
    uintptr_t e = 0, f = 0;
    if (hooks::Readable((void*)(base + vt), kVtSlots * 8) &&
        hooks::TryRead((void*)(base + vt + kExecuteSlot * 8), &e, 8) &&
        hooks::TryRead((void*)(base + vt + kFinishSlot * 8), &f, 8) && e == base + exec && f == base + kTaskFinish)
        return true;
    Log("MVR G-buffer: %s vtable does not match (Execute at AMS2AVX.exe+0x%llX, Finish +0x%llX; expected +0x%llX, "
        "+0x%llX); %s",
        name, (unsigned long long)(e ? e - base : 0), (unsigned long long)(f ? f - base : 0), (unsigned long long)exec,
        (unsigned long long)kTaskFinish, fatal ? "off" : "no early SSAO and lights (mvr_early)");
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
        if (CheckVtable(base, kSsaoVtable, kSsaoExecute, "CSSAORender", false) &&
            CheckVtable(base, kPhase2Vtable, kPhase2Execute, "CPhase2Render", false) &&
            CheckVtable(base, kPhase3Vtable, kPhase3Execute, "CPhase3Render", false)) {
            g_ssaoVt = base + kSsaoVtable;
            g_phase2Vt = base + kPhase2Vtable;
            g_phase3Vt = base + kPhase3Vtable;
        }
        g_early = Cfg().mvrEarly;
        g_forwardOn = Cfg().mvrForward && g_phase3Vt;
        SetForwardDepth(Cfg().mvrForwardDepth);
        g_velSkipOn = Cfg().mvrVelocitySkip;
        g_gbufPsOn = Cfg().mvrGbufferPs;
        g_directOn = Cfg().mvrDirectExtract;
        g_nativeOn = Cfg().mvrNativeTargets;
        g_msColorSkip = Cfg().mvrMsColorSkip;
        g_msSlices = Cfg().mvrMsSlices;
        g_installed = true;
        Log("MVR G-buffer: on; outer L's and outer R's G-buffer pass draw the eye's focus too (MVR, view 1 in the "
            "corner of slice 1 of two-slice copies), the focus groups skip CPhase1BuildCommandThread and CPhase1Render "
            "and get it in their clones; jitter fix %s; hotkey 0x%X+0x%X; early SSAO and lights of focus L (mvr_early) "
            "%s, hotkey 0x%X+0x%X; forward test in shots (mvr_forward_test) %s; live forward of both eyes "
            "(mvr_forward) %s, hotkey 0x%X+0x%X, its depth %s (hotkey 0x%X+0x%X)",
            Cfg().mvrJitterFix ? "on" : "off", Cfg().mvrKeyMod, Cfg().mvrKey,
            !g_ssaoVt ? "unavailable" : EarlyName(g_early), Cfg().mvrKeyMod, Cfg().mvrEarlyKey,
            !g_phase3Vt ? "unavailable" : Cfg().mvrForwardTest ? "on" : "off",
            !g_phase3Vt ? "unavailable" : g_forwardOn ? "on" : "off", Cfg().mvrKeyMod, Cfg().mvrForwardKey,
            ForwardDepthName(g_fwdDepth), Cfg().mvrKeyMod, Cfg().mvrForwardDepthKey);
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
        // Clones made again (B17's arrays, a list change): the module's views on the old ones go (they hold them).
        if (const uint32_t cg = clones::Generation(); cg != g_cloneGen) {
            g_cloneGen = cg;
            g_cloneViews.clear();
            g_srvClones.clear();
        }
        // Copies not used for a while (B17: the game's targets became arrays, or the other mode) are released.
        for (int k = 0; k < (int)g_copies.size(); ++k)
            if (g_copies[k].arr && g_copies[k].lastFrame + 120 < g_frame) {
                DropViewsOf(k);
                g_copies[k] = Copy{};
            }
        if (g_shot.armed) {
            ReportShot();
            g_shot = Shot{};
        }
        if (g_fshot.armed) {
            ReportForward();
            g_fshot = FwdShot{};
        }
        t_fwdFocusTask = nullptr;
        g_probe = Probe{};
        g_curFocus = -1;
        g_velBound.store(-1, std::memory_order_relaxed);
        const bool on = enabled && On();
        if (on != g_enabled)
            Log("MVR G-buffer: %s (frame %llu)", on ? "ON" : "OFF (the focus groups run their own pass)",
                (unsigned long long)g_frame);
        g_enabled = on;
        for (FocusState& f : g_focus) f.decided = f.skip = f.early = f.earlyLights = f.ssaoAgain = f.refill = false;
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
        if (Cfg().mvrForwardTest && g_early) {
            const char* why = g_early != 1      ? "it needs mvr_early=1 (focus L's SSAO and lights in outer L's group)"
                              : !mvrps::On()    ? "it needs [debug] mvr_ps_patch=1"
                              : !g_phase3Vt     ? "the CPhase3Render vtable did not match"
                              : max < 2 * kMaxTargets || maxPairs < kMaxTargets ? "no room for its images"
                                                                               : nullptr;
            if (why) {
                Log("MVR forward test: no comparison in this shot (%s)", why);
                return;
            }
            FwdShot& sh = g_fshot;
            sh = FwdShot{};
            sh.armed = true;
            sh.base = base;
            sh.frame = g_frame;
            for (int k = 0; k < kMaxTargets; ++k) {
                snprintf(g_fwdNames[2 * k], sizeof(g_fwdNames[0]), "mvr_f%d_focus", k);
                snprintf(g_fwdNames[2 * k + 1], sizeof(g_fwdNames[0]), "mvr_f%d_ours", k);
                names[base + 2 * k] = g_fwdNames[2 * k];
                names[base + 2 * k + 1] = g_fwdNames[2 * k + 1];
                snprintf(g_fwdLabels[k], sizeof(g_fwdLabels[k]), "MVR forward %d: focus L vs view 1 of outer L", k);
                pairs[k] = {base + 2 * k, base + 2 * k + 1, g_fwdLabels[k]};
            }
            *pairCount = kMaxTargets;
            n = 2 * kMaxTargets;
            Log("MVR forward test: armed in frame %llu (outer L's CPhase3Render with MVR, focus L's own for the "
                "comparison)",
                (unsigned long long)g_frame);
            return;
        }
        if (!Cfg().mvrShotCompare) {
            Log("MVR G-buffer: no comparison in this shot (mvr_shot_compare=0: the shot shows the frame as it runs)");
            return;
        }
        if (g_early) {  // the comparison needs focus's own G-buffer pass, the early tasks need it skipped
            Log("MVR G-buffer: no comparison in this shot (mvr_early %s: the shot shows the early tasks)",
                EarlyName(g_early));
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
    if (g_fshot.armed) Protect("CancelShot", [&] { g_fshot = FwdShot{}; });
}

void SetWindow(int eye, bool valid, double sx, double ox, double sy, double oy, uint32_t width, uint32_t height) {
    if (eye < 0 || eye > 1) return;
    Win& w = g_win[eye];
    if (w.frame != g_frame) g_winPrev[eye] = w;
    w.valid = valid;
    w.frame = g_frame;
    w.sx = sx;
    w.ox = ox;
    w.sy = sy;
    w.oy = oy;
    w.w = width;
    w.h = height;
}

bool MotionWindows(int eye, double cur[4], double prev[4], bool* fromOuter) {
    if (eye < 0 || eye > 1 || !On()) return false;
    const Win& w = g_win[eye];
    if (!w.valid || w.frame != g_frame) return false;
    const Win& p = g_winPrev[eye].valid && g_winPrev[eye].frame + 1 == g_frame ? g_winPrev[eye] : w;
    const double c[4] = {w.sx, w.ox, w.sy, w.oy}, q[4] = {p.sx, p.ox, p.sy, p.oy};
    memcpy(cur, c, sizeof(c));
    memcpy(prev, q, sizeof(q));
    *fromOuter = g_fwdSkipFrame[eye] == g_frame;
    return true;
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

void SetForward(bool on) { g_forwardOn = on; }
bool GetForward() { return g_forwardOn; }
void SetForwardDepth(int mode) { g_fwdDepth = mode >= 0 && mode < kForwardDepthModes ? mode : 3; }
int GetForwardDepth() { return g_fwdDepth; }
const char* ForwardDepthName(int mode) {
    switch (mode) {
    case 0: return "AS DRAWN (view 1's depth as MVR drew it)";
    case 1: return "BIAS (plus mvr_depth_bias float steps)";
    case 2: return "STASH (conservative like the G-buffer stash: dilate, slope, bias)";
    case 3: return "AS DRAWN + VELOCITY (as drawn; focus's velocity pass on a conservative copy)";
    }
    return "?";
}
void SetMsColorSkip(bool on) { g_msColorSkip = on; }
bool GetMsColorSkip() { return g_msColorSkip; }
void SetPhase3BuildVtable(uintptr_t vt) { g_phase3BuildVt = vt; }

void FocusGroupIn(int eye) {
    if (!On() || eye < 0 || eye > 1) return;
    g_curFocus = eye;
}

Kind KindOf(uint8_t* task) {
    uintptr_t vt = 0;
    if (!g_installed || !task || !hooks::TryRead(task, &vt, 8) || !vt) return Kind::Other;
    if (vt == g_phase1Vt) return Kind::Phase1;
    if (vt == g_buildVt) return Kind::Build;
    if (vt == g_ssaoVt) return Kind::Ssao;
    if (vt == g_phase2Vt) return Kind::Phase2;
    if (vt == g_phase3Vt) return Kind::Phase3;
    return Kind::Other;
}

void SetEarly(int mode) { g_early = mode >= 0 && mode < kEarlyModes ? mode : 0; }
int GetEarly() { return g_early; }
const char* EarlyName(int mode) {
    switch (mode) {
    case 1: return "SSAO and lights";
    case 2: return "SSAO only";
    case 3: return "DIAGNOSIS swap and CBeginEye only (no task moved)";
    case 4: return "DIAGNOSIS SSAO early and again in the focus group";
    case 5: return "DIAGNOSIS SSAO early, the G-buffer filled again in the focus group";
    }
    return "off";
}

void EarlyRefused(int eye, const char* why) {
    if (eye < 0 || eye > 1 || !why) return;
    AddReason(g_st.earlyReasons, why);
    AddReason(g_tot.earlyReasons, why);
}

bool EarlyWanted(int eye) {
    // Focus L; focus R only with the live forward pass (B9d), into focus R's own clones (mvr_pass.h).
    if (!On() || !g_early || !g_enabled || eye < 0 || eye > 1) return false;
    if (eye == 1 && !(g_forwardOn && g_early == 1)) return false;
    if (!g_ssaoVt || !g_phase2Vt || !g_phase3Vt) return false;      // logged at the start
    const char* why = nullptr;
    if (g_mode != Mode::Skip)
        why = "a diagnosis mode of the G-buffer is on";
    else if (g_shot.armed)
        why = "shot frame";
    else if (!g_stash[eye].valid || g_stash[eye].frame != g_frame)
        why = "no stash from outer's pass in this frame";
    else if (g_focus[eye].decided)
        why = "the focus group already decided";
    else if (eye == 1 && !TwinsReady())
        why = "focus R's own clones are not made yet";
    if (why) EarlyRefused(eye, why);
    return why == nullptr;
}

bool EarlyBegin(int eye, bool projected) {
    if (eye < 0 || eye > 1) return false;
    bool ok = false;
    Protect("EarlyBegin", [&] {
        FocusGroup(eye, projected);
        if (const char* why = Decide(eye)) {
            EarlyRefused(eye, why);
            return;
        }
        FocusState& f = g_focus[eye];
        f.decided = f.skip = f.early = true;
        f.earlyLights = g_early == 1;
        f.ssaoAgain = g_early == 4;
        f.refill = g_early == 5;
        if (g_lastSkip[eye] == 0) {
            ++g_st.switches[eye];
            ++g_tot.switches[eye];
        }
        g_lastSkip[eye] = 1;
        g_track = g_forwardOn;  // the clones the early tasks write: each gets focus R's twin
        Fill(Imm(), eye);
        ok = true;
    });
    return ok;
}

void EarlyWaited(double ms) {
    for (Stats* s : {&g_st, &g_tot}) {
        ++s->earlyWaits;
        if (ms > 0.001) ++s->earlyWaited;
        s->earlyWaitMs += ms;
        s->earlyWaitMax = std::max(s->earlyWaitMax, ms);
    }
}

void EarlyEnd(int eye, double ms, bool ok) {
    g_track = false;
    if (eye < 0 || eye > 1) return;
    ++g_st.earlyRuns[eye];
    ++g_tot.earlyRuns[eye];
    g_st.earlyMs[eye] += ms;
    g_tot.earlyMs[eye] += ms;
    if (ok && g_focus[eye].early && g_focus[eye].earlyLights) g_earlyOk[eye] = g_frame;
    if (!ok) {
        ++g_st.earlyFaults;
        ++g_tot.earlyFaults;
    }
}

void TaskMapBegin(const char* group, const char* name, bool detail) {
    g_map = TaskMap{};
    g_map.on = true;
    g_map.detail = detail;
    snprintf(g_map.group, sizeof(g_map.group), "%s", group ? group : "?");
    snprintf(g_map.name, sizeof(g_map.name), "%s", name ? name : "?");
}

void TaskMapEnd() {
    if (!g_map.on) return;
    g_map.on = false;
    if (!g_map.draws && !g_map.binds && !g_map.clears) return;
    std::string det;
    if (g_map.detail) {
        auto list = [](std::vector<std::pair<UINT, void*>>& v, const char* pre, bool desc) {
            std::sort(v.begin(), v.end());
            std::string out;
            for (const auto& [slot, r] : v) {
                char b[96];
                std::string d;
                if (desc) {
                    bool ok = false;
                    const D3D11_TEXTURE2D_DESC td = DescOf((ID3D11Resource*)r, &ok);
                    if (ok) d = " " + std::to_string(td.Width) + "x" + std::to_string(td.Height) + " f" +
                                std::to_string((int)td.Format);
                }
                snprintf(b, sizeof(b), "%s%s%u%s %s", out.empty() ? "" : ", ", pre, slot, d.c_str(), ResId(r).c_str());
                out += b;
            }
            return out;
        };
        det = "; PS srv: " + list(g_map.srvs, "t", true) + (g_map.srvMore ? ", more" : "") + "; PS cb: " +
              list(g_map.cbs, "b", false);
    }
    Log("MVR task map (frame %llu): %s %s: draws %u, binds %u, clears %u; targets: %s%s%s", (unsigned long long)g_frame,
        g_map.group, g_map.name, g_map.draws, g_map.binds, g_map.clears,
        g_map.targets.empty() ? "none" : g_map.targets.c_str(), g_map.more ? ", more" : "", det.c_str());
}

// Forward test (B9c): outer L's CPhase3Render starts the MVR pass, focus L's is noted for the comparison after it.
void ForwardTask(uint8_t* task, int group) {
    FwdShot& sh = g_fshot;
    if (group == kFocusL) {
        if (sh.outerDone && !sh.focusDone) t_fwdFocusTask = task;
        return;
    }
    if (group != kOuterL || sh.outerDone) return;
    const char* why = !g_enabled                    ? "MVR G-buffer off (hotkey)"
                      : g_mode != Mode::Skip        ? "a diagnosis mode of the G-buffer is on"
                      : g_earlyOk[0] != g_frame     ? "focus L's SSAO and lights did not run early in this frame"
                      : g_p.part != PartNone        ? "another MVR pass is running"
                      : !mvrps::On()                ? "the PS patch is off"
                                                    : nullptr;
    if (why) return FwdRefused(why);
    PassBegin(task, kOuterL, true);
}

// B9d: why outer's live forward pass does not run now (null: it runs; "": off, nothing counted).
const char* LiveWanted(int e) {
    if (!g_forwardOn) return "";
    if (!g_enabled) return "MVR G-buffer off (hotkey)";
    if (g_mode != Mode::Skip) return "a diagnosis mode of the G-buffer is on";
    if (g_earlyOk[e] != g_frame) return "focus's SSAO and lights did not run early in this frame";
    if (g_p.part != PartNone) return "another MVR pass is running";
    if (!mvrps::On()) return "the PS patch is off";
    return nullptr;
}

// CPhase3Render (build = false) or focus's CPhase3BuildCommandThread (build = true) of an eye group. Outer's starts the
// forward test (B9c, eye L in a forward test shot) or the live forward pass (B9d); a focus group's is skipped (true)
// when outer's live pass put view 1 into its clones this frame, the build only together with the render.
bool Phase3Task(uint8_t* task, int group, bool build) {
    if (group <= kOuterR) {
        if (build) return false;
        if (g_fshot.armed && group == kOuterL) {
            ForwardTask(task, group);
            return false;
        }
        const char* why = LiveWanted(group);
        if (!why)
            PassBegin(task, group, true, true);
        else if (*why)
            FwdReason(group, why);
        return false;
    }
    const int eye = group - kFocusL;
    if (g_fshot.armed && eye == 0 && !build) ForwardTask(task, group);  // the test's comparison: focus L's own pass
    if (!g_forwardOn) return false;
    const bool ok = g_fwdOk[eye] == g_frame && g_focus[eye].early;
    if (build) {
        if (ok) {
            ++g_fs.buildSkipped[eye];
            ++g_ftot.buildSkipped[eye];
            ++g_testLive.buildSkipped;
            return true;
        }
        g_buildRan[eye] = g_frame;
        return false;
    }
    if (ok && g_buildRan[eye] != g_frame) {
        ++g_fs.skipped[eye];
        ++g_ftot.skipped[eye];
        ++g_testLive.focusSkipped;
        g_fwdSkipFrame[eye] = g_frame;
        return true;
    }
    if (ok) FwdReason(eye, "focus's CPhase3BuildCommandThread ran before outer's forward pass was done");
    ++g_fs.own[eye];
    ++g_ftot.own[eye];
    return false;
}

bool BeforeTask(uint8_t* task, int group) {
    if (!On() || !task || group < kOuterL || group > kFocusR) return false;
    uintptr_t vt = 0;
    if (!hooks::TryRead(task, &vt, 8) || !vt) return false;
    if (g_fshot.armed && (vt == g_phase1Vt || (g_phase3Vt && vt == g_phase3Vt))) {
        g_probe = Probe{};
        g_probe.on = true;
        g_probe.group = group;
        g_probe.task = task;
        g_probe.name = vt == g_phase1Vt ? "CPhase1Render" : "CPhase3Render";
    }
    const bool build3 = g_phase3BuildVt && vt == g_phase3BuildVt;
    if ((g_phase3Vt && vt == g_phase3Vt) || build3) {
        bool skip3 = false;
        Protect("BeforeTask", [&] { skip3 = Phase3Task(task, group, build3); });
        if (skip3 && g_probe.task == task) g_probe = Probe{};
        return skip3;
    }
    const bool late = vt == g_ssaoVt || vt == g_phase2Vt;  // after the G-buffer pair: skipped only after an early run
    if (vt != g_phase1Vt && vt != g_buildVt && !late) return false;
    bool skip = false;
    Protect("BeforeTask", [&] {
        if (group <= kOuterR) {
            if (vt == g_phase1Vt) PassBegin(task, group, false);
            return;
        }
        const int eye = group - kFocusL;
        FocusState& f = g_focus[eye];
        if (late) {
            if (f.decided && f.early && (vt == g_ssaoVt ? !f.ssaoAgain : f.earlyLights)) {
                ++g_st.earlySkipped[eye];
                ++g_tot.earlySkipped[eye];
                skip = true;
            }
            return;
        }
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
                // An early run filled the clones before focus's SSAO and lights (mode 5: again, diagnosis).
                if (!f.early || f.refill) Fill(Imm(), eye);
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
    if (skip && g_probe.task == task) g_probe = Probe{};  // a skipped task draws nothing
    return skip;
}

void AfterTask(uint8_t* task) {
    if (!task) return;
    if (g_probe.on && task == g_probe.task) {
        if (!g_probe.done)
            Log("MVR jitter probe (frame %llu): %s %s drew nothing with VS slot 12 bound", (unsigned long long)g_frame,
                kGroupName[g_probe.group], g_probe.name);
        g_probe = Probe{};
    }
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
    if (task == t_fwdFocusTask) {
        t_fwdFocusTask = nullptr;
        Protect("AfterTask", [&] { ShotForwardFocus(Imm()); });
        return;
    }
    if (g_p.part == PartNone || task != g_p.task) return;
    Protect("AfterTask", [&] { PassEnd(Imm()); });
}

bool MapTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv, ID3D11DepthStencilView* dsv,
                ID3D11RenderTargetView** rtvOut, ID3D11DepthStencilView** dsvOut) {
    MapBind(c, n, rtv, dsv);
    if (g_track && !t_internal && c == Imm()) {  // B9d: an early run's writes into focus's clones
        for (UINT i = 0; i < n && rtv; ++i) NoteWritten(rtv[i]);
        NoteWritten(dsv);
    }
    if (!t_internal && c == Imm()) g_velBound.store(-1, std::memory_order_relaxed);
    // B9d mode 3: focus's velocity pass draws on the conservative copy of the forward depth.
    if (g_fwdDepth == 3 && g_forwardOn && !t_internal && c == Imm() && dsv && n && rtv && rtv[0] &&
        g_p.part == PartNone) {
        ID3D11DepthStencilView* vel = nullptr;
        Protect("MapTargets (velocity)", [&] { vel = VelocityDsv(rtv[0], dsv); });
        if (vel) {
            for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) rtvOut[i] = rtv[i];
            *dsvOut = vel;
            ++g_fs.velBinds;
            ++g_ftot.velBinds;
            ++g_testLive.velBinds;
            g_velBound.store(g_curFocus, std::memory_order_relaxed);  // B11: its draws go through VelocityDraw
            return true;
        }
    }
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
        if (any && !outer) {
            if (p.forward) {  // a target of another size: MVR pauses, the game's views go as they are
                if (!p.paused) ++p.pauses;
                p.paused = true;
                if (p.views != 1 && SetViews(c, 1)) p.views = 1;
                return;
            }
            EndMvr(c, false);  // the masks: the G-buffer is done
            NoteTail(n, rtv, dsv);
            return;
        }
        if (any) p.paused = false;
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
        if (p.forward) {  // their content goes in before the next draw (a clear first makes it unneeded)
            for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) MarkPending(rtvOut[i]);
            MarkPending(*dsvOut);
            // B12 step 5: view 1's slice for the bound copies (all of them must agree).
            NvU32 want = 0;
            bool split = false;
            auto slice = [&](ID3D11View* v) {
                const int k = v ? CopyOfView(v) : -1;
                if (k < 0) return;
                const NvU32 s = SliceFor(g_copies[k]);
                split |= want && s != want;
                want = s;
            };
            for (UINT i = 0; i < n && i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) slice(rtvOut[i]);
            slice(*dsvOut);
            if (split) PassFailed(p, "targets bound together with view 1 in different slices");
            if (want && want != p.viewSlice) {
                p.viewSlice = want;
                if (p.views != 1) SetViews(c, p.views);
            }
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

// B9d: a clear of a target of another size in the live forward pass, for its focus clone at the end.
void RecordClear(ID3D11View* v, bool depth, const FLOAT* col, UINT flags, FLOAT d, UINT8 s) {
    if (!g_p.live || g_otherClears.size() >= 16) return;
    OtherClear oc;
    ComPtr<ID3D11Resource> r;
    v->GetResource(&r);
    oc.orig = g_cloneFn ? r.Get() : (ID3D11Resource*)clones::Original(r.Get());
    oc.depth = depth;
    if (depth)
        static_cast<ID3D11DepthStencilView*>(v)->GetDesc(&oc.dd);
    else
        static_cast<ID3D11RenderTargetView*>(v)->GetDesc(&oc.rd);
    if (col) memcpy(oc.col, col, sizeof(oc.col));
    oc.flags = flags;
    oc.d = d;
    oc.s = s;
    g_otherClears.push_back(oc);
}

// B12 step 5: a clear of a three-slice copy in the forward pass goes to slice 0 and to view 1's slice only; the third
// slice keeps the other eye's focus. `game`: the view the game clears (its own view of the array, or the copy's view of
// all slices that the pass read back from the bound state). Returns the view of slice 0 alone for the game's call (view
// 1's slice is cleared here), or null: clear `game` as it is.
template <class Desc>
bool SliceDesc(Desc& d, UINT slice);
template <>
bool SliceDesc(D3D11_RENDER_TARGET_VIEW_DESC& d, UINT slice) {
    switch (d.ViewDimension) {
    case D3D11_RTV_DIMENSION_TEXTURE2D: {
        const UINT mip = d.Texture2D.MipSlice;
        d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {mip, slice, 1};
        return true;
    }
    case D3D11_RTV_DIMENSION_TEXTURE2DARRAY: d.Texture2DArray.FirstArraySlice = slice; d.Texture2DArray.ArraySize = 1; return true;
    case D3D11_RTV_DIMENSION_TEXTURE2DMS: d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY; d.Texture2DMSArray = {slice, 1}; return true;
    case D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY: d.Texture2DMSArray = {slice, 1}; return true;
    default: return false;
    }
}
template <>
bool SliceDesc(D3D11_DEPTH_STENCIL_VIEW_DESC& d, UINT slice) {
    switch (d.ViewDimension) {
    case D3D11_DSV_DIMENSION_TEXTURE2D: {
        const UINT mip = d.Texture2D.MipSlice;
        d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {mip, slice, 1};
        return true;
    }
    case D3D11_DSV_DIMENSION_TEXTURE2DARRAY: d.Texture2DArray.FirstArraySlice = slice; d.Texture2DArray.ArraySize = 1; return true;
    case D3D11_DSV_DIMENSION_TEXTURE2DMS: d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY; d.Texture2DMSArray = {slice, 1}; return true;
    case D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY: d.Texture2DMSArray = {slice, 1}; return true;
    default: return false;
    }
}

ID3D11View* ClearOwnSlices(ID3D11DeviceContext* c, ID3D11View* m, ID3D11View* game, const FLOAT* col, UINT flags,
                           FLOAT d, UINT8 st) {
    const int k = CopyOfView(m);
    if (k < 0 || !g_copies[k].native || g_copies[k].slices < 3 || !g_copies[k].arr) return nullptr;
    Copy& cp = g_copies[k];
    ComPtr<ID3D11Resource> gr;
    game->GetResource(&gr);
    if (gr.Get() != (ID3D11Resource*)cp.arr.Get()) return nullptr;
    const UINT slice = SliceFor(cp);
    ID3D11View *one = nullptr, *zero = nullptr;
    if (!cp.depth) {
        D3D11_RENDER_TARGET_VIEW_DESC r1{}, r0{};
        static_cast<ID3D11RenderTargetView*>(game)->GetDesc(&r1);
        r0 = r1;
        if (!SliceDesc(r1, slice) || !SliceDesc(r0, 0)) return nullptr;
        one = CloneViewOf(cp.arr.Get(), 0, &r1, sizeof(r1));
        zero = CloneViewOf(cp.arr.Get(), 0, &r0, sizeof(r0));
    } else {
        D3D11_DEPTH_STENCIL_VIEW_DESC d1{}, d0{};
        static_cast<ID3D11DepthStencilView*>(game)->GetDesc(&d1);
        d0 = d1;
        if (!SliceDesc(d1, slice) || !SliceDesc(d0, 0)) return nullptr;
        one = CloneViewOf(cp.arr.Get(), 1, &d1, sizeof(d1));
        zero = CloneViewOf(cp.arr.Get(), 1, &d0, sizeof(d0));
    }
    if (!one || !zero) return nullptr;
    {
        Internal in;
        clones::RawScope raw;
        if (cp.depth)
            c->ClearDepthStencilView(static_cast<ID3D11DepthStencilView*>(one), flags, d, st);
        else
            c->ClearRenderTargetView(static_cast<ID3D11RenderTargetView*>(one), col);
    }
    ++g_p.sliceClears;
    return zero;
}

ID3D11RenderTargetView* MapClear(ID3D11DeviceContext* c, ID3D11RenderTargetView* v, const FLOAT col[4]) {
    if (g_map.on && !t_internal && c == Imm()) ++g_map.clears;
    if (g_track && !t_internal && c == Imm()) NoteWritten(v);
    StartAtClear(c, v);
    if (!InPass(c) || !v) return v;
    if (g_p.forward && !OuterSize(v)) {
        ++g_p.pausedClears;
        Protect("ClearRenderTargetView", [&] { RecordClear(v, false, col, 0, 0, 0); });
        return v;
    }
    ID3D11RenderTargetView* out = v;
    Protect("ClearRenderTargetView", [&] {
        ++g_p.clears;
        if (ID3D11RenderTargetView* m = ArrayRtv(c, v)) {
            out = m;
            if (g_p.forward) {
                FillCopy(c, m, true);  // B12: the clear covers both slices
                // B12 step 5: not the other eye's slice
                if (ID3D11View* z = ClearOwnSlices(c, m, v, col, 0, 0, 0)) out = static_cast<ID3D11RenderTargetView*>(z);
            }
        }
    });
    return out;
}

ID3D11DepthStencilView* MapClear(ID3D11DeviceContext* c, ID3D11DepthStencilView* v, UINT flags, FLOAT depth,
                                 UINT8 stencil) {
    if (g_map.on && !t_internal && c == Imm()) ++g_map.clears;
    if (g_track && !t_internal && c == Imm()) NoteWritten(v);
    StartAtClear(c, v);
    if (!InPass(c) || !v) return v;
    if (g_p.forward && !OuterSize(v)) {
        ++g_p.pausedClears;
        Protect("ClearDepthStencilView", [&] { RecordClear(v, true, nullptr, flags, depth, stencil); });
        return v;
    }
    ID3D11DepthStencilView* out = v;
    Protect("ClearDepthStencilView", [&] {
        ++g_p.clears;
        if (ID3D11DepthStencilView* m = ArrayDsv(c, v)) {
            out = m;
            // First: a clear may cover only the depth or only the stencil (B12: one of all of it needs no copy-in).
            if (g_p.forward) {
                D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
                m->GetDesc(&dd);
                const bool hasStencil = dd.Format == DXGI_FORMAT_D24_UNORM_S8_UINT ||
                                        dd.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
                FillCopy(c, m, (flags & D3D11_CLEAR_DEPTH) && (!hasStencil || (flags & D3D11_CLEAR_STENCIL)));
                if (ID3D11View* z = ClearOwnSlices(c, m, v, nullptr, flags, depth, stencil))  // B12 step 5
                    out = static_cast<ID3D11DepthStencilView*>(z);
            }
        }
    });
    return out;
}

bool BeforeDraw(ID3D11DeviceContext* c) {
    if (g_probe.on && !g_probe.done && !t_internal && c == Imm() && g_g.dev)
        Protect("ProbeCamera", [&] { ProbeCamera(c); });
    if (g_map.on && !t_internal && c == Imm()) {
        ++g_map.draws;
        if (g_map.detail) MapDetail(c);
    }
    if (g_p.part == PartTail && !t_internal && c == Imm()) {
        ++g_p.tailDraws;
        if (g_p.lateBound) ++g_p.lateDraws;
    }
    if (!t_pass || t_internal || c != Imm()) return false;
    if (g_p.forward) {
        Protect("Draw", [&] {
            FlushPending(c);
            ForwardDraw(c);
        });
        return false;
    }
    if (g_gbufPsOn) Protect("Draw", [&] { GbufDraw(c); });
    if (!g_shot.armed || g_p.eye != 0) return false;
    ++g_p.draws;
    if (g_p.views != 2) ++g_p.plainDraws;
    return false;
}

void AfterDraw(ID3D11DeviceContext*) {}

// B11: the linear depth among the PS shader resources of the velocity pass's first draw (its camera motion pass reads
// it): an R32 float view of a plain w x h texture. Null when none or more than one distinct texture.
static ComPtr<ID3D11Texture2D> LinearDepthOfDraw(ID3D11DeviceContext* c, UINT w, UINT h) {
    ID3D11ShaderResourceView* srv[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT] = {};
    c->PSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, srv);
    ComPtr<ID3D11Texture2D> found;
    bool twice = false;
    for (ID3D11ShaderResourceView* v : srv) {
        if (!v) continue;
        D3D11_SHADER_RESOURCE_VIEW_DESC vd{};
        v->GetDesc(&vd);
        ComPtr<ID3D11Resource> r;
        v->GetResource(&r);
        ComPtr<ID3D11Texture2D> t;
        if (vd.Format == DXGI_FORMAT_R32_FLOAT && vd.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D && r &&
            SUCCEEDED(r.As(&t))) {
            D3D11_TEXTURE2D_DESC d{};
            t->GetDesc(&d);
            if (d.Width == w && d.Height == h && d.ArraySize == 1 && d.SampleDesc.Count == 1) {
                if (found && found.Get() != t.Get()) twice = true;
                found = t;
            }
        }
        v->Release();
    }
    return twice ? nullptr : found;
}

// B11: a draw while the eye's velocity pass is bound. The first one of the frame decides: the fill source writes the
// vectors into the bound target (then this and the pass's later draws are dropped) or the pass draws as before.
static bool VelocityDraw(ID3D11DeviceContext* c, int e) {
    VelSkip& v = g_velSkip[e];
    if (v.frame != g_frame) {
        v.frame = g_frame;
        v.state = VelUndecided;
    }
    if (v.state == VelUndecided) {
        v.state = VelDraws;
        const char* why = nullptr;
        if (!g_velSkipOn) {
            why = "";  // off: no reason line
        } else if (!g_velFill) {
            why = "no fill source";
        } else {
            ComPtr<ID3D11RenderTargetView> rtv;
            c->OMGetRenderTargets(1, &rtv, nullptr);
            ComPtr<ID3D11Resource> r;
            if (rtv) rtv->GetResource(&r);
            ComPtr<ID3D11Texture2D> target;
            if (r) r.As(&target);
            if (!target) {
                why = "no velocity target bound";
            } else {
                D3D11_TEXTURE2D_DESC td{};
                target->GetDesc(&td);
                const ComPtr<ID3D11Texture2D> depth = LinearDepthOfDraw(c, td.Width, td.Height);
                if (!depth) {
                    why = "no linear depth (R32 at the target's size) at the pass's first draw";
                } else {
                    Internal in;
                    if (g_velFill(c, e, depth.Get(), target.Get(), &why)) v.state = VelFilled;
                }
            }
        }
        if (v.state == VelFilled) {
            ++g_fs.velFilled[e];
            ++g_ftot.velFilled[e];
            ++g_testLive.velFilled;
        } else {
            ++g_fs.velDrawn[e];
            ++g_ftot.velDrawn[e];
            ++g_testLive.velDrawn;
            if (why && *why) {
                char b[200];
                snprintf(b, sizeof(b), "%s %s", kEye[e], why);
                AddReason(g_fs.velReasons, b);
                AddReason(g_ftot.velReasons, b);
            }
        }
    }
    if (v.state != VelFilled) return false;
    ++g_fs.velSkippedDraws;
    ++g_ftot.velSkippedDraws;
    ++g_testLive.velSkippedDraws;
    return true;
}

bool SkipDraw(ID3D11DeviceContext* c) {
    const int e = g_velBound.load(std::memory_order_relaxed);
    if (e < 0 || e > 1 || t_internal || c != Imm()) return false;
    bool skip = false;
    Protect("SkipDraw", [&] { skip = VelocityDraw(c, e); });
    return skip;
}

void SetVelocityFill(VelocityFillFn fn) { g_velFill = fn; }
void SetVelocitySkip(bool on) { g_velSkipOn = on; }
bool GetVelocitySkip() { return g_velSkipOn; }
void SetGbufferPs(bool on) { g_gbufPsOn = on; }
bool GetGbufferPs() { return g_gbufPsOn; }
void SetDirectExtract(bool on) { g_directOn = on; }
bool GetDirectExtract() { return g_directOn; }
void SetNativeTargets(bool on) {
    // The views found for the game's views point to copies of the other mode: found again (CopyFor) at their next bind.
    if (on != g_nativeOn) g_views.clear();
    g_nativeOn = on;
}
bool GetNativeTargets() { return g_nativeOn; }

std::string Describe() {
    if (!g_installed) return "MVR G-buffer off";
    char buf[256];
    snprintf(buf, sizeof(buf),
             "MVR G-buffer %s (focus pass from outer: L %llu, R %llu of %llu frames; forward from outer: L %llu, R %llu)",
             g_dead ? "STOPPED" : !On() ? "off (hooks)" : g_enabled ? "on" : "off (hotkey)",
             (unsigned long long)g_tot.skipped[0], (unsigned long long)g_tot.skipped[1],
             (unsigned long long)g_tot.frames, (unsigned long long)g_ftot.skipped[0],
             (unsigned long long)g_ftot.skipped[1]);
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
        "buffers %llu), mixed binds %llu, late binds %llu (draws after them %llu), map failures %llu, SetMultiViewMode failures %llu; "
        "patched PS with view 1's rows (mvr_gbuffer_ps %s) %llu draws, kept as before (screen textures) %llu, live "
        "track maps from outer %llu (failed %llu); slice 1 straight into focus's clones (mvr_direct_extract %s) L %llu "
        "/ R %llu targets, through the stash L %llu / R %llu, direct writes failed %llu; the game's own two-slice "
        "arrays (mvr_native_targets %s) G-buffer %llu / forward %llu targets, copied in and back G-buffer %llu / "
        "forward %llu",
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
        (unsigned long long)s.viewsFailures, g_gbufPsOn ? "on" : "off", (unsigned long long)s.gbufPsDraws,
        (unsigned long long)s.gbufPsKept, (unsigned long long)s.liveTracks, (unsigned long long)s.liveTrackFails,
        g_directOn ? "on" : "off", (unsigned long long)s.direct[0], (unsigned long long)s.direct[1],
        (unsigned long long)s.viaStash[0], (unsigned long long)s.viaStash[1], (unsigned long long)s.directFails,
        g_nativeOn ? "on" : "off", (unsigned long long)s.nativeT[0], (unsigned long long)s.nativeT[1],
        (unsigned long long)s.copiedT[0], (unsigned long long)s.copiedT[1]);
    if (g_early || s.earlyRuns[0] || s.earlyRuns[1] || !s.earlyReasons.empty()) {
        std::string er;
        for (const Reason& r : s.earlyReasons) {
            char b[200];
            snprintf(b, sizeof(b), "%s%s %llu", er.empty() ? "" : "; ", r.text.c_str(), (unsigned long long)r.n);
            er += b;
        }
        auto ems = [&](int e) { return s.earlyRuns[e] ? s.earlyMs[e] / s.earlyRuns[e] : 0.0; };
        Log("MVR early SSAO and lights%s (%llu frames, %s): runs in outer's group L %llu / R %llu, CPU per run L %.3f / "
            "R %.3f ms, engine faults %llu; focus tasks skipped L %llu / R %llu; waits for the recording workers %llu "
            "(non-zero %llu, avg %.3f, max %.3f ms)%s%s%s",
            totals ? " (session)" : "", (unsigned long long)s.frames, EarlyName(g_early),
            (unsigned long long)s.earlyRuns[0], (unsigned long long)s.earlyRuns[1], ems(0), ems(1),
            (unsigned long long)s.earlyFaults, (unsigned long long)s.earlySkipped[0],
            (unsigned long long)s.earlySkipped[1], (unsigned long long)s.earlyWaits, (unsigned long long)s.earlyWaited,
            s.earlyWaits ? s.earlyWaitMs / s.earlyWaits : 0.0, s.earlyWaitMax, er.empty() ? "" : "; not run: ",
            er.c_str(), "");
    }
    FwdStats& f = totals ? g_ftot : g_fs;
    if (g_forwardOn || f.passes[0] || f.passes[1] || !f.reasons.empty()) {
        std::string fr;
        for (const Reason& r : f.reasons) {
            char b[240];
            snprintf(b, sizeof(b), "%s%s %llu", fr.empty() ? "" : "; ", r.text.c_str(), (unsigned long long)r.n);
            fr += b;
        }
        auto fms = [&](int e) { return f.passes[e] ? f.ms[e] / f.passes[e] : 0.0; };
        std::string vr;
        for (const Reason& r : f.velReasons) {
            char b[240];
            snprintf(b, sizeof(b), "%s%s %llu", vr.empty() ? "" : "; ", r.text.c_str(), (unsigned long long)r.n);
            vr += b;
        }
        size_t twins = 0;
        for (ID3D11Resource* o : g_twinOrigs) twins += !g_cloneFn && clones::HasTwin(o);
        Log("MVR forward%s (%llu frames, %s, depth %d): outer passes L %llu / R %llu (into focus's clones L %llu / R %llu), "
            "task CPU per pass L %.3f / R %.3f ms, draws per pass %.0f; focus forward skipped L %llu / R %llu (builds "
            "%llu / %llu), own L %llu / R %llu; focus clones written from outer %zu, with focus R's twin %zu; "
            "other-size clears repeated %llu (shared "
            "%llu); velocity pass binds on the conservative depth %llu; velocity passes with outer's vectors and no draws "
            "(mvr_velocity_skip %s) L %llu / R %llu, draws skipped %llu, passes drawn L %llu / R %llu%s%s; live track "
            "maps from outer %llu (failed %llu); targets: copied in from focus's clones %llu, cleared first (no copy-in) "
            "%llu, multisampled (per-sample copies) %llu, multisampled colors without copy-in after a depth clear "
            "(mvr_ms_color_skip %s) %llu, read by focus in their slice (B12 step 5, mvr_ms_slices %s) L %llu / R %llu "
            "with %llu clears of only their own slices%s%s",
            totals ? " (session)" : "", (unsigned long long)s.frames, g_forwardOn ? "on" : "off", g_fwdDepth,
            (unsigned long long)f.passes[0], (unsigned long long)f.passes[1], (unsigned long long)f.ok[0],
            (unsigned long long)f.ok[1], fms(0), fms(1),
            f.passes[0] + f.passes[1] ? (double)f.draws / (f.passes[0] + f.passes[1]) : 0.0,
            (unsigned long long)f.skipped[0], (unsigned long long)f.skipped[1], (unsigned long long)f.buildSkipped[0],
            (unsigned long long)f.buildSkipped[1], (unsigned long long)f.own[0], (unsigned long long)f.own[1],
            g_twinOrigs.size(), twins, (unsigned long long)f.otherClears,
            (unsigned long long)f.otherClearMisses, (unsigned long long)f.velBinds, g_velSkipOn ? "on" : "off",
            (unsigned long long)f.velFilled[0], (unsigned long long)f.velFilled[1],
            (unsigned long long)f.velSkippedDraws, (unsigned long long)f.velDrawn[0], (unsigned long long)f.velDrawn[1],
            vr.empty() ? "" : " (", vr.empty() ? "" : (vr + ")").c_str(), (unsigned long long)f.liveTracks,
            (unsigned long long)f.liveTrackFails, (unsigned long long)f.copyIns, (unsigned long long)f.clearFills,
            (unsigned long long)f.msTargets, g_msColorSkip ? "on" : "off", (unsigned long long)f.colorSkips,
            g_msSlices ? "on" : "off", (unsigned long long)f.sliceTargets[0], (unsigned long long)f.sliceTargets[1],
            (unsigned long long)f.sliceClears, fr.empty() ? "" : "; not from outer: ", fr.c_str());
    }
    if (!totals) {
        s = Stats{};
        g_fs = FwdStats{};
    }
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

void TestNative(NativeFn isArray, WantArrayFn want, SliceFn slice) {
    g_nativeFn = isArray;
    g_wantFn = want;
    g_sliceFn = slice;
}

TestState TestGet() { return g_test; }

void TestDepthBias(int steps, float slope, bool dilate) {
    g_testBias = steps;
    g_testSlope = slope;
    g_testDilate = dilate;
}
ID3D11Texture2D* TestArray(int k) { return k >= 0 && k < (int)g_copies.size() ? g_copies[k].arr.Get() : nullptr; }

void TestForward(const void* phase3Vtable) {
    g_phase3Vt = (uintptr_t)phase3Vtable;
    g_fshot = FwdShot{};
    g_fshot.armed = true;
    g_fshot.frame = g_frame;
    g_earlyOk[0] = g_frame;
}

TestForwardState TestForwardGet() {
    TestForwardState t;
    const FwdShot& sh = g_fshot;
    const Pass& p = sh.pass;
    t.outerDone = sh.outerDone;
    t.focusDone = sh.focusDone;
    t.failed = p.failed;
    t.why = p.failed ? p.why : sh.why;
    t.draws = p.draws;
    t.plainDraws = p.plainDraws;
    t.psPatchedDraws = p.psPatchedDraws;
    t.psCbRebinds = p.psCbRebinds;
    t.srvMisses = p.srvMisses;
    t.copyIns = p.copyIns;
    t.binds = p.binds;
    t.mapFailures = p.mapFailures;
    t.viewsFailures = p.viewsFailures;
    t.copiedBack = p.copiedBack;
    t.pauses = p.pauses;
    t.pausedDraws = p.pausedDraws;
    t.pausedClears = p.pausedClears;
    t.stashed = sh.n;
    return t;
}

ID3D11Texture2D* TestForwardStash(int k) { return k >= 0 && k < g_fwdN ? g_fwd[k].tex.Get() : nullptr; }

void TestForwardLive(const void* phase3Vtable) {
    g_phase3Vt = (uintptr_t)phase3Vtable;
    g_forwardOn = true;
    g_fshot = FwdShot{};
    g_testLive = TestLive{};
    g_early = 1;
    g_earlyOk[0] = g_frame;
    FocusState& f = g_focus[0];
    f.decided = f.skip = f.early = f.earlyLights = true;
}

void TestEarlyMark(int eye) {
    if (eye < 0 || eye > 1) return;
    FocusState& f = g_focus[eye];
    f.decided = f.skip = f.early = f.earlyLights = true;
    g_earlyOk[eye] = g_frame;
}

TestLiveState TestLiveGet() {
    TestLiveState t;
    const TestLive& s = g_testLive;
    t.passes = s.passes;
    t.ok = s.ok;
    t.focusSkipped = s.focusSkipped;
    t.buildSkipped = s.buildSkipped;
    t.otherClears = s.otherClears;
    t.otherClearMisses = s.otherClearMisses;
    t.twinEntries = (unsigned)g_twinOrigs.size();
    t.velBinds = s.velBinds;
    t.velFilled = s.velFilled;
    t.velSkippedDraws = s.velSkippedDraws;
    t.velDrawn = s.velDrawn;
    t.native = s.native;
    t.copiedBack = s.copiedBack;
    t.copyIns = s.copyIns;
    t.clearFills = s.clearFills;
    t.msTargets = s.msTargets;
    t.colorSkips = s.colorSkips;
    t.sliceTargets = s.sliceTargets;
    t.sliceClears = s.sliceClears;
    t.why = s.why;
    return t;
}

void TestEnd() {
    const bool faulted = g_test.faulted;
    g_test = TestState{};
    g_test.faulted = faulted;
}

}  // namespace qvfr::mvrpass
