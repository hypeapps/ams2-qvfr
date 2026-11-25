#include "sun_fx.h"

#include <intrin.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "clones.h"
#include "common.h"
#include "dxbc.h"
#include "gpu.h"
#include "hook_prof.h"
#include "hooks.h"
#include "mvr_pass.h"
#include "sr_cb.h"
#include "sr_probe.h"
#include "sun_fx_ps.h"  // kSunFxPS (fxc from shaders/sun_fx.hlsl)
#include "sun_fx_vs.h"  // kSunFxVS
#include "taa_views.h"
#include "windscreen.h"

namespace qvfr::sunfx {

namespace {

// Effect pixel shaders by the DXBC checksum (bytes 4..19 of the bytecode as four little-endian dwords, the "Shader hash"
// RenderDoc shows), AMS2AVX.exe of the reference build (capture 2026-11-05, docs/ams2-renderer.md).
struct Known {
    uint32_t sum[4];
    Kind kind;
    const char* name;
};
const Known kKnown[] = {
    {{0x8df2fc6f, 0x36155ac1, 0x9db80c63, 0x453230e5}, KindRays, "crepuscular rays to screen"},
    {{0x82975810, 0x569f4c4b, 0xafbdb398, 0x3d613fba}, KindFlare, "lens flare sprite"},
};
constexpr int kKnownCount = (int)(sizeof(kKnown) / sizeof(kKnown[0]));

// ID3D11Device / ID3D11DeviceContext vtable slots (d3d11.h order).
constexpr int kSlotCreatePixelShader = 15;
constexpr int kSlotPSSetShader = 9;
constexpr int kSlotDrawIndexed = 12;
constexpr int kSlotDraw = 13;

using CreatePSFn = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                               ID3D11PixelShader**);
using PSSetShaderFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11PixelShader*, ID3D11ClassInstance* const*,
                                               UINT);
using DrawIndexedFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, INT);
using DrawFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT);
CreatePSFn o_createPs = nullptr;
PSSetShaderFn o_psSet = nullptr;
DrawIndexedFn o_drawIndexed = nullptr;
DrawFn o_draw = nullptr;
void* g_target[4] = {};  // hooked functions: CreatePixelShader, PSSetShader, DrawIndexed, Draw
bool g_installed = false;
std::atomic<DrawReplaceFn> g_drawReplace{nullptr};  // DLAA in quad mode (SetDrawReplace)
std::atomic<InsideFn> g_drawInside{nullptr};
std::atomic<AfterFn> g_drawAfter{nullptr};

// Effect shader objects: open addressing, written under g_tableM, read without a lock (kind before pointer). Entries
// stay; a shader object made at a recorded address from other bytecode sets the entry's kind to none.
// Effect shaders and the car paint shaders of the bonnet reflection (with their texture slot in g_slotArg).
constexpr int kSlots = 1024;
std::atomic<void*> g_slotPtr[kSlots];
std::atomic<int> g_slotKind[kSlots];
std::atomic<uint8_t> g_slotArg[kSlots];
std::mutex g_tableM;
std::atomic<int> g_entries{0};
std::atomic<uint64_t> g_created[kKnownCount];  // shader objects made from each known bytecode
std::atomic<bool> g_createdLogged[kKnownCount];

size_t Slot(const void* p) { return (size_t)((((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull) >> 54); }

Kind Lookup(const void* p, uint8_t* arg = nullptr) {
    if (!p || !g_entries.load(std::memory_order_relaxed)) return KindNone;
    size_t i = Slot(p);
    for (int n = 0; n < kSlots; ++n, i = (i + 1) & (kSlots - 1)) {
        void* s = g_slotPtr[i].load(std::memory_order_acquire);
        if (!s) return KindNone;
        if (s == p) {
            const Kind k = (Kind)g_slotKind[i].load(std::memory_order_acquire);
            if (arg) *arg = g_slotArg[i].load(std::memory_order_relaxed);
            return k;
        }
    }
    return KindNone;
}

void SetKind(void* p, Kind kind, uint8_t arg = 0) {
    std::lock_guard lock(g_tableM);
    size_t i = Slot(p);
    for (int n = 0; n < kSlots; ++n, i = (i + 1) & (kSlots - 1)) {
        void* s = g_slotPtr[i].load(std::memory_order_relaxed);
        if (s == p) {
            g_slotArg[i].store(arg, std::memory_order_relaxed);
            g_slotKind[i].store(kind, std::memory_order_release);
            return;
        }
        if (!s) {
            if (kind == KindNone) return;
            g_slotArg[i].store(arg, std::memory_order_relaxed);
            g_slotKind[i].store(kind, std::memory_order_relaxed);
            g_slotPtr[i].store(p, std::memory_order_release);
            g_entries.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}

int Classify(const void* code, SIZE_T len) {
    if (!code || len < 20 || memcmp(code, "DXBC", 4) != 0) return -1;
    uint32_t sum[4];
    memcpy(sum, (const uint8_t*)code + 4, sizeof(sum));
    for (int k = 0; k < kKnownCount; ++k)
        if (!memcmp(sum, kKnown[k].sum, sizeof(sum))) return k;
    return -1;
}

bool IsEffect(Kind k) { return k == KindRays || k == KindFlare; }

// The car paint shaders of the bonnet reflection (sun_fx.h SetBonnetFromOuter): bonnetReflectionAndClamp of
// COMMON_BUFFER2 used and motionBlurMap_tex bound. Their texture slot, or -1. Reflection data as fxc writes it for
// shader model 5.0 (bindings 32 bytes, 5.1: 40; cbuffers 24; variables 40, flag 2 = used).
int BonnetSlot(const void* code, SIZE_T len) {
    static const char kVar[] = "bonnetReflectionAndClamp";
    const char* b = (const char*)code;
    if (!code || std::search(b, b + len, kVar, kVar + sizeof(kVar) - 1) == b + len) return -1;
    std::vector<dxbc::Chunk> chunks;
    if (!dxbc::Parse(code, len, chunks, nullptr)) return -1;
    const dxbc::Chunk* rdef = dxbc::Find(chunks, dxbc::FourCC('R', 'D', 'E', 'F'));
    if (!rdef || rdef->data.size() < 28) return -1;
    const std::vector<uint8_t>& d = rdef->data;
    auto get = [&](size_t o) {
        uint32_t v = 0;
        if (o + 4 <= d.size()) memcpy(&v, d.data() + o, 4);
        return v;
    };
    auto named = [&](uint32_t off, const char* name) {
        const size_t n = strlen(name);
        return off < d.size() && d.size() - off > n && !memcmp(d.data() + off, name, n + 1);
    };
    const uint32_t cbCount = get(0), cbOff = get(4), bindCount = get(8), bindOff = get(12), target = get(16);
    const uint32_t major = (target >> 8) & 0xFF, minor = target & 0xFF;
    if (major != 5) return -1;
    const size_t bindStride = minor >= 1 ? 40 : 32;
    if (cbOff + 24ull * cbCount > d.size() || bindOff + bindStride * bindCount > d.size()) return -1;
    bool used = false;
    for (uint32_t i = 0; i < cbCount && !used; ++i) {
        const size_t c = cbOff + 24ull * i;
        const uint32_t varCount = get(c + 4), varOff = get(c + 8);
        if (varOff + 40ull * varCount > d.size()) return -1;
        for (uint32_t v = 0; v < varCount; ++v) {
            const size_t e = varOff + 40ull * v;
            if (named(get(e), kVar) && (get(e + 12) & 2)) used = true;
        }
    }
    if (!used) return -1;
    for (uint32_t i = 0; i < bindCount; ++i) {
        const size_t e = bindOff + bindStride * i;
        if (get(e + 4) == 2 /* texture */ && named(get(e), "motionBlurMap_tex") && get(e + 24) == 1 &&
            get(e + 20) < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)
            return (int)get(e + 20);
    }
    return -1;
}

// ---------------------------------------------------------------- render thread state

// Accumulation of the effects of one outer eye: what its effect draws added, at outer's HDR size.
struct Accum {
    ID3D11Texture2D* tex = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    UINT w = 0, h = 0;
    ID3D11RenderTargetView* checkedFor = nullptr;  // the HDR view whose size was checked in this group (only compared)
    bool failed = false;
};

struct Pipeline {
    ID3D11Device* dev = nullptr;  // referenced
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11Buffer* cb = nullptr;
    ID3D11SamplerState* sampler = nullptr;
    ID3D11BlendState* blend = nullptr;
    ID3D11DepthStencilState* dss = nullptr;
    ID3D11RasterizerState* rs = nullptr;
    bool failed = false;
};

struct Stats {
    uint64_t frames = 0;          // frames with the module on
    uint64_t outerFrames = 0;     // frames in which both outer eyes drew effects
    uint64_t replays = 0;         // outer effect draws repeated into the accumulation
    uint64_t skipped = 0;         // focus effect draws skipped
    uint64_t composites = 0;      // focus passes at a focus effect draw
    uint64_t anchorComposites = 0;  // focus passes at the learned anchor (no effect draw in the focus group)
    uint64_t missed = 0;          // focus groups without the pass although outer drew effects
    uint64_t anchorNoHdr = 0;     // anchor draws that did not read the HDR target
    uint64_t bonnetOuter = 0;     // outer's car paint draws (their image noted)
    uint64_t bonnetFocus = 0;     // focus's car paint draws given outer's image
    uint64_t bonnetMissed = 0;    // focus's car paint draws without outer's image from this frame (the game's kept)
};

std::atomic<ID3D11DeviceContext*> g_imm{nullptr};
std::atomic<int> g_group{-1};  // eye group running now (render thread), -1 = none
bool g_enabled = false;
std::atomic<bool> g_dead{false};  // a fault, or the game's device is not the hooked one: off for the session
bool g_ctxHooked = false;  // the context of g_imm calls the hooked functions (CheckContext; CurrentPs)
bool g_testing = false;
uint32_t g_gen = 0;
ID3D11DeviceContext* g_checkedCtx = nullptr;
Pipeline g_p;
Accum g_acc[2];
bool g_valid[2] = {};     // this frame's outer eye drew effects into its accumulation
void* g_hdrOrig[2] = {};  // the engine's HDR texture behind them (clones::Original, only compared)
bool g_done = false;      // focus group: the pass ran
float g_origin[2] = {0, 0}, g_scale[2] = {1, 1};
// Current pixel shader and render target 0 of the immediate context.
ID3D11PixelShader* g_curPs = nullptr;
Kind g_curKind = KindNone;
uint8_t g_curArg = 0;  // KindBonnet: the shader's motionBlurMap_tex slot
// O1 step 3 (CurrentPs): g_curPs's epoch (the latest NewPsEpoch when the hook saw it set) and the epoch now.
uint32_t g_curPsEpoch = 0, g_psEpoch = 1;
uint64_t g_psReads = 0;
ID3D11RenderTargetView* g_curRt0 = nullptr;
// Anchor learning in outer: after an effect draw, the first draw into another target is the anchor's pixel shader.
ID3D11RenderTargetView* g_sunRtv = nullptr;
bool g_afterSun = false;
ID3D11PixelShader* g_anchorPs = nullptr;  // only compared
Stats g_st, g_total;
// Bonnet reflection: the view outer of each eye bound for its car paint in the frame g_bonnetFrame (referenced).
bool g_bonnetOn = true;
ID3D11ShaderResourceView* g_bonnetSrv[2] = {};
uint64_t g_bonnetFrame[2] = {};
uint64_t g_frameNo = 0;
std::atomic<uint64_t> g_bonnetMade{0};
bool g_missLogged = false, g_noHdrLogged = false;

template <class T>
void SafeRelease(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

void Fail(const char* where, unsigned long code) {
    if (g_dead.exchange(true)) return;
    Log("Sun effects: ERROR: exception 0x%08lX in %s; off for this session (the game draws its own effects in the "
        "focus views)", code, where);
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

void ReleaseObjects() {
    for (Accum& a : g_acc) {
        SafeRelease(a.srv);
        SafeRelease(a.rtv);
        SafeRelease(a.tex);
        a = Accum{};
    }
    SafeRelease(g_p.vs);
    SafeRelease(g_p.ps);
    SafeRelease(g_p.cb);
    SafeRelease(g_p.sampler);
    SafeRelease(g_p.blend);
    SafeRelease(g_p.dss);
    SafeRelease(g_p.rs);
    SafeRelease(g_p.dev);
    g_p = Pipeline{};
    g_valid[0] = g_valid[1] = false;
    SafeRelease(g_bonnetSrv[0]);
    SafeRelease(g_bonnetSrv[1]);
}

bool EnsurePipeline(ID3D11Device* dev) {
    if (g_p.dev != dev) {
        ReleaseObjects();
        g_p.dev = dev;
        dev->AddRef();
    }
    if (g_p.failed) return false;
    if (g_p.vs) return true;
    HRESULT hr = dev->CreateVertexShader(kSunFxVS, sizeof(kSunFxVS), nullptr, &g_p.vs);
    if (SUCCEEDED(hr)) hr = dev->CreatePixelShader(kSunFxPS, sizeof(kSunFxPS), nullptr, &g_p.ps);
    if (SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC b{};
        b.ByteWidth = 32;
        b.Usage = D3D11_USAGE_DYNAMIC;
        b.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        b.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = dev->CreateBuffer(&b, nullptr, &g_p.cb);
    }
    if (SUCCEEDED(hr)) {
        D3D11_SAMPLER_DESC s{};
        s.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        s.AddressU = s.AddressV = s.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        s.ComparisonFunc = D3D11_COMPARISON_NEVER;
        s.MaxLOD = D3D11_FLOAT32_MAX;
        hr = dev->CreateSamplerState(&s, &g_p.sampler);
    }
    if (SUCCEEDED(hr)) {
        D3D11_BLEND_DESC b{};
        D3D11_RENDER_TARGET_BLEND_DESC& t = b.RenderTarget[0];
        t.BlendEnable = TRUE;
        t.SrcBlend = t.DestBlend = D3D11_BLEND_ONE;
        t.BlendOp = D3D11_BLEND_OP_ADD;
        t.SrcBlendAlpha = D3D11_BLEND_ZERO;
        t.DestBlendAlpha = D3D11_BLEND_ONE;
        t.BlendOpAlpha = D3D11_BLEND_OP_ADD;
        t.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN |
                                  D3D11_COLOR_WRITE_ENABLE_BLUE;
        hr = dev->CreateBlendState(&b, &g_p.blend);
    }
    if (SUCCEEDED(hr)) {
        D3D11_DEPTH_STENCIL_DESC d{};
        d.DepthEnable = FALSE;
        d.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        d.DepthFunc = D3D11_COMPARISON_ALWAYS;
        hr = dev->CreateDepthStencilState(&d, &g_p.dss);
    }
    if (SUCCEEDED(hr)) {
        D3D11_RASTERIZER_DESC r{};
        r.FillMode = D3D11_FILL_SOLID;
        r.CullMode = D3D11_CULL_NONE;
        r.DepthClipEnable = TRUE;
        hr = dev->CreateRasterizerState(&r, &g_p.rs);
    }
    if (FAILED(hr)) {
        ReleaseObjects();
        g_p.failed = true;
        Log("Sun effects: cannot create the shaders and states (0x%08lX); off for this device", (unsigned long)hr);
        return false;
    }
    Log("Sun effects: pass ready (vs %zu B, ps %zu B)", sizeof(kSunFxVS), sizeof(kSunFxPS));
    return true;
}

// The accumulation of outer eye `e` in the size of the bound HDR target `rtv`.
bool EnsureAccum(int e, ID3D11RenderTargetView* rtv) {
    Accum& a = g_acc[e];
    if (a.tex && a.checkedFor == rtv) return true;
    ID3D11Resource* res = nullptr;
    rtv->GetResource(&res);
    ID3D11Texture2D* t = nullptr;
    if (res) res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    SafeRelease(res);
    if (!t) return false;
    D3D11_TEXTURE2D_DESC hd{};
    t->GetDesc(&hd);
    t->Release();
    if (a.tex && (a.w != hd.Width || a.h != hd.Height)) {
        SafeRelease(a.srv);
        SafeRelease(a.rtv);
        SafeRelease(a.tex);
        a.failed = false;
    }
    if (!a.tex) {
        if (a.failed && a.w == hd.Width && a.h == hd.Height) return false;
        a.w = hd.Width;
        a.h = hd.Height;
        D3D11_TEXTURE2D_DESC d{};
        d.Width = hd.Width;
        d.Height = hd.Height;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.SampleDesc = {1, 0};
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        HRESULT hr = g_p.dev->CreateTexture2D(&d, nullptr, &a.tex);
        if (SUCCEEDED(hr)) hr = g_p.dev->CreateRenderTargetView(a.tex, nullptr, &a.rtv);
        if (SUCCEEDED(hr)) hr = g_p.dev->CreateShaderResourceView(a.tex, nullptr, &a.srv);
        if (FAILED(hr)) {
            SafeRelease(a.srv);
            SafeRelease(a.rtv);
            SafeRelease(a.tex);
            a.failed = true;
            Log("Sun effects: cannot create the %s accumulation %u x %u (0x%08lX)", e ? "right" : "left", d.Width,
                d.Height, (unsigned long)hr);
            return false;
        }
        Log("Sun effects: %s eye accumulation %u x %u (outer HDR target format %d, %u samples)", e ? "right" : "left",
            d.Width, d.Height, (int)hd.Format, hd.SampleDesc.Count);
    }
    a.checkedFor = rtv;
    return true;
}

void* OriginalOf(ID3D11View* v) {
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    void* o = r ? clones::Original(r) : nullptr;
    SafeRelease(r);
    return o;
}

// Outer: the game's effect draw, then the same draw into the eye's accumulation.
template <class D>
void ReplayOuter(ID3D11DeviceContext* c, int e, D&& draw) {
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
    const bool ok = rtv[0] && EnsureAccum(e, rtv[0]);
    draw();
    if (ok) {
        Accum& a = g_acc[e];
        if (!g_valid[e]) {
            const float zero[4] = {};
            c->ClearRenderTargetView(a.rtv, zero);
            g_valid[e] = true;
            g_hdrOrig[e] = OriginalOf(rtv[0]);
        }
        c->OMSetRenderTargets(1, &a.rtv, nullptr);
        draw();
        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
        ++g_st.replays;
        g_sunRtv = rtv[0];
        g_afterSun = true;
    }
    for (ID3D11RenderTargetView*& v : rtv) SafeRelease(v);
    SafeRelease(dsv);
}

// The state the pass changes besides the targets and viewports.
struct PassState {
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
};

// Adds accumulation `e` into the bound target over the bound viewport 0.
bool Pass(ID3D11DeviceContext* c, int e) {
    Accum& a = g_acc[e];
    if (!a.srv) return false;
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    if (!nvp || vp.Width <= 0 || vp.Height <= 0) return false;
    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(c->Map(g_p.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return false;
    const float k[8] = {g_origin[0], g_origin[1], g_scale[0], g_scale[1], vp.TopLeftX, vp.TopLeftY, 1 / vp.Width,
                        1 / vp.Height};
    memcpy(m.pData, k, sizeof(k));
    c->Unmap(g_p.cb, 0);
    PassState s;
    c->IAGetInputLayout(&s.il);
    c->IAGetPrimitiveTopology(&s.topo);
    c->VSGetShader(&s.vs, nullptr, nullptr);
    c->HSGetShader(&s.hs, nullptr, nullptr);
    c->DSGetShader(&s.ds, nullptr, nullptr);
    c->GSGetShader(&s.gs, nullptr, nullptr);
    c->PSGetShader(&s.ps, nullptr, nullptr);
    c->PSGetConstantBuffers(0, 1, &s.cb);
    c->PSGetShaderResources(0, 1, &s.srv);
    c->PSGetSamplers(0, 1, &s.sampler);
    c->OMGetBlendState(&s.bs, s.blendFactor, &s.sampleMask);
    c->OMGetDepthStencilState(&s.dss, &s.stencilRef);
    c->RSGetState(&s.rs);

    c->IASetInputLayout(nullptr);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(g_p.vs, nullptr, 0);
    c->HSSetShader(nullptr, nullptr, 0);
    c->DSSetShader(nullptr, nullptr, 0);
    c->GSSetShader(nullptr, nullptr, 0);
    c->PSSetShader(g_p.ps, nullptr, 0);
    c->PSSetConstantBuffers(0, 1, &g_p.cb);
    c->PSSetShaderResources(0, 1, &a.srv);
    c->PSSetSamplers(0, 1, &g_p.sampler);
    const float one[4] = {1, 1, 1, 1};
    c->OMSetBlendState(g_p.blend, one, 0xFFFFFFFF);
    c->OMSetDepthStencilState(g_p.dss, 0);
    c->RSSetState(g_p.rs);
    o_draw(c, 3, 0);

    c->IASetInputLayout(s.il);
    c->IASetPrimitiveTopology(s.topo);
    c->VSSetShader(s.vs, nullptr, 0);
    c->HSSetShader(s.hs, nullptr, 0);
    c->DSSetShader(s.ds, nullptr, 0);
    c->GSSetShader(s.gs, nullptr, 0);
    c->PSSetShader(s.ps, nullptr, 0);
    c->PSSetConstantBuffers(0, 1, &s.cb);
    c->PSSetShaderResources(0, 1, &s.srv);
    c->PSSetSamplers(0, 1, &s.sampler);
    c->OMSetBlendState(s.bs, s.blendFactor, s.sampleMask);
    c->OMSetDepthStencilState(s.dss, s.stencilRef);
    c->RSSetState(s.rs);
    SafeRelease(s.il);
    SafeRelease(s.vs);
    SafeRelease(s.hs);
    SafeRelease(s.ds);
    SafeRelease(s.gs);
    SafeRelease(s.ps);
    SafeRelease(s.cb);
    SafeRelease(s.srv);
    SafeRelease(s.sampler);
    SafeRelease(s.bs);
    SafeRelease(s.dss);
    SafeRelease(s.rs);
    return true;
}

// Focus without an effect draw: at the anchor (a pass that reads the HDR target), the pass into the HDR target first.
bool PassAtAnchor(ID3D11DeviceContext* c, int e) {
    constexpr UINT kSrv = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
    ID3D11ShaderResourceView* srv[kSrv] = {};
    c->PSGetShaderResources(0, kSrv, srv);
    ID3D11ShaderResourceView* hdr = nullptr;
    for (ID3D11ShaderResourceView* v : srv)
        if (v && !hdr && OriginalOf(v) == g_hdrOrig[e]) hdr = v;
    bool ok = false;
    ID3D11RenderTargetView* target = nullptr;
    ID3D11Resource* res = nullptr;
    if (hdr) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
        hdr->GetDesc(&sd);
        hdr->GetResource(&res);
        D3D11_RENDER_TARGET_VIEW_DESC rd{};
        rd.Format = sd.Format;
        if (sd.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2DMS) {
            rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
        } else if (sd.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D) {
            rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            rd.Texture2D.MipSlice = sd.Texture2D.MostDetailedMip;
        } else {
            SafeRelease(res);
        }
        if (res) g_p.dev->CreateRenderTargetView(res, &rd, &target);
    }
    ID3D11Texture2D* t = nullptr;
    if (target && res) res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t);
    if (t) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        ID3D11DepthStencilView* dsv = nullptr;
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
        D3D11_VIEWPORT vps[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
        UINT nvp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        c->RSGetViewports(&nvp, vps);
        ID3D11ShaderResourceView* none[kSrv] = {};
        c->PSSetShaderResources(0, kSrv, none);  // the HDR texture is read here: unbound before it becomes the target
        c->OMSetRenderTargets(1, &target, nullptr);
        const D3D11_VIEWPORT full = {0, 0, (float)d.Width, (float)d.Height, 0, 1};
        c->RSSetViewports(1, &full);
        ok = Pass(c, e);
        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
        c->RSSetViewports(nvp, vps);
        c->PSSetShaderResources(0, kSrv, srv);
        for (ID3D11RenderTargetView*& v : rtv) SafeRelease(v);
        SafeRelease(dsv);
        t->Release();
    }
    SafeRelease(target);
    SafeRelease(res);
    for (ID3D11ShaderResourceView*& v : srv) SafeRelease(v);
    if (!hdr) {
        ++g_st.anchorNoHdr;
        if (!g_noHdrLogged) {
            g_noHdrLogged = true;
            Log("Sun effects: the anchor draw of a focus group does not read outer's HDR target (counted from now on)");
        }
    }
    return ok;
}

enum Action { Forward, Skip, Done };

// Render thread, a draw on the game's immediate context inside an eye group with the module on.
template <class D>
Action OnDraw(ID3D11DeviceContext* c, int g, D&& draw) {
    const Kind k = g_curKind;
    if (g < 2) {
        if (IsEffect(k)) {
            ReplayOuter(c, g, draw);
            return Done;
        }
        if (g_afterSun && g_curRt0 != g_sunRtv) {
            g_anchorPs = g_curPs;
            g_afterSun = false;
        }
        return Forward;
    }
    const int e = g - 2;
    if (IsEffect(k)) {
        ++g_st.skipped;
        if (g_valid[e] && !g_done && Pass(c, e)) {
            g_done = true;
            ++g_st.composites;
        }
        return Skip;
    }
    if (!g_done && g_valid[e] && g_anchorPs && g_curPs == g_anchorPs && PassAtAnchor(c, e)) {
        g_done = true;
        ++g_st.anchorComposites;
    }
    return Forward;
}

bool Active(ID3D11DeviceContext* c, int& g) {
    g = g_group.load(std::memory_order_relaxed);
    return g >= 0 && g_enabled && c == g_imm.load(std::memory_order_relaxed) && !g_dead.load(std::memory_order_relaxed);
}

// A car paint draw of the bonnet reflection about to be forwarded: outer notes the image it reads, focus gets outer's
// (the game's view in *keep, put back by BonnetAfter). The slot bound, or -1.
int BonnetBefore(ID3D11DeviceContext* c, ID3D11ShaderResourceView** keep) {
    if (g_curKind != KindBonnet || !g_bonnetOn) return -1;
    const int g = g_group.load(std::memory_order_relaxed);
    if (g < 0 || c != g_imm.load(std::memory_order_relaxed) || g_dead.load(std::memory_order_relaxed)) return -1;
    int bound = -1;
    Protect("BonnetBefore", [&] {
        const UINT slot = g_curArg;
        if (g < 2) {
            ID3D11ShaderResourceView* v = nullptr;
            c->PSGetShaderResources(slot, 1, &v);
            if (v != g_bonnetSrv[g]) {
                SafeRelease(g_bonnetSrv[g]);
                g_bonnetSrv[g] = v;
            } else {
                SafeRelease(v);
            }
            g_bonnetFrame[g] = g_frameNo;
            ++g_st.bonnetOuter;
            return;
        }
        const int e = g - 2;
        if (!g_bonnetSrv[e] || g_bonnetFrame[e] != g_frameNo) {
            ++g_st.bonnetMissed;
            return;
        }
        c->PSGetShaderResources(slot, 1, keep);
        clones::RawScope raw;  // outer's view as it is (the bind fix would send it to focus's clone)
        c->PSSetShaderResources(slot, 1, &g_bonnetSrv[e]);
        ++g_st.bonnetFocus;
        bound = (int)slot;
    });
    return bound;
}

void BonnetAfter(ID3D11DeviceContext* c, int slot, ID3D11ShaderResourceView* keep) {
    Protect("BonnetAfter", [&] {
        clones::RawScope raw;  // the game's view was translated when it was bound
        c->PSSetShaderResources((UINT)slot, 1, &keep);
    });
    SafeRelease(keep);
}

// ---------------------------------------------------------------- detours

std::atomic<CreatePsFilterFn> g_createPsFilter{nullptr};

HRESULT STDMETHODCALLTYPE D_CreatePixelShader(ID3D11Device* dev, const void* code, SIZE_T len,
                                              ID3D11ClassLinkage* link, ID3D11PixelShader** out) {
    const CreatePsFilterFn filter = g_createPsFilter.load(std::memory_order_relaxed);
    HRESULT hr = filter ? filter(dev, code, len, link, out, o_createPs, (uintptr_t)_ReturnAddress())
                        : o_createPs(dev, code, len, link, out);
    if (SUCCEEDED(hr) && out && *out) {
        Protect("CreatePixelShader", [&] {
            taaviews::OnPixelShaderMade(*out, code, len);  // the game's TAA resolve (focus motion fix)
            windscreen::OnPixelShaderMade(*out, code, len);  // the sharpen and the windscreen layers
            const int k = Classify(code, len);
            if (k >= 0) {
                SetKind(*out, kKnown[k].kind);
                g_created[k].fetch_add(1, std::memory_order_relaxed);
                if (!g_createdLogged[k].exchange(true))
                    Log("Sun effects: the game made its \"%s\" pixel shader (%p)", kKnown[k].name, (void*)*out);
            } else if (const int slot = BonnetSlot(code, len); slot >= 0) {
                SetKind(*out, KindBonnet, (uint8_t)slot);
                if (g_bonnetMade.fetch_add(1, std::memory_order_relaxed) == 0)
                    Log("Sun effects: the game made its first car paint pixel shader with the bonnet reflection (%p, "
                        "image in t%d)", (void*)*out, slot);
            } else if (Lookup(*out) != KindNone) {
                SetKind(*out, KindNone);  // a new shader at the address of a released effect or car paint shader
            }
        });
    }
    return hr;
}

void STDMETHODCALLTYPE D_PSSetShader(ID3D11DeviceContext* c, ID3D11PixelShader* ps, ID3D11ClassInstance* const* ci,
                                     UINT n) {
    hookprof::Scope prof(hookprof::KPsShader);
    {
        hookprof::Orig o;
        o_psSet(c, ps, ci, n);
    }
    if (c != g_imm.load(std::memory_order_relaxed)) return;
    windscreen::OnPixelShaderSet(ps);
    g_curPs = ps;
    g_curKind = Lookup(ps, &g_curArg);
    g_curPsEpoch = g_psEpoch;
}

struct IndexedArgs {
    ID3D11DeviceContext* c;
    UINT count, start;
    INT base;
};
void DrawIndexedTail(void* p);

void STDMETHODCALLTYPE D_DrawIndexed(ID3D11DeviceContext* c, UINT count, UINT start, INT base) {
    hookprof::Scope prof(hookprof::KDraw);
    if (const InsideFn inside = g_drawInside.load(std::memory_order_relaxed); inside && inside()) {
        hookprof::Orig o;
        o_drawIndexed(c, count, start, base);
        return;
    }
    if (count == 6 && mvrpass::TaskMapOn()) mvrpass::TaskMapDraw6(c, _ReturnAddress());  // S1 probe
    srcb::OnDraw(c);  // S2.1: after the TAA resolve of a DLSS SR group, the constants at the view size
    if (srprobe::Armed()) srprobe::OnDraw(c, _ReturnAddress(), "DrawIndexed");  // S2.1 probe frame only
    if (mvrpass::SkipDraw(c)) return;  // B11: a focus velocity pass whose target got outer's vectors
    IndexedArgs args{c, count, start, base};
    // The passes after TAA: the game's sharpen and the windscreen glass (windscreen.h; the rest of the detour as `draw`).
    if (windscreen::OnDraw(c, DrawIndexedTail, &args)) return;
    DrawIndexedTail(&args);
}

void DrawIndexedTail(void* p) {
    const IndexedArgs& args = *static_cast<IndexedArgs*>(p);
    ID3D11DeviceContext* c = args.c;
    const UINT count = args.count, start = args.start;
    const INT base = args.base;
    if (const DrawReplaceFn replace = g_drawReplace.load(std::memory_order_relaxed)) {
        hookprof::SubScope sp(hookprof::SDlss);
        if (replace(c, count)) return;  // DLSS replaced the game's TAA resolve
    }
    bool taaFix;
    {
        hookprof::SubScope sp(hookprof::STaaDraw);
        taaFix = taaviews::OnDraw(c, true, count);
    }
    const bool mvr = mvrpass::BeforeDraw(c);
    int g = -1;
    Action a = Forward;
    if (Active(c, g)) {
        Protect("DrawIndexed", [&] {
            a = OnDraw(c, g, [&] {
                hookprof::Orig o;
                o_drawIndexed(c, count, start, base);
            });
        });
    }
    if (a == Forward) {
        ID3D11ShaderResourceView* keep = nullptr;
        const int bonnet = BonnetBefore(c, &keep);
        {
            hookprof::Orig o;
            o_drawIndexed(c, count, start, base);
        }
        if (bonnet >= 0) BonnetAfter(c, bonnet, keep);
    }
    if (mvr) mvrpass::AfterDraw(c);
    if (taaFix) taaviews::AfterDraw(c);
    if (const AfterFn after = g_drawAfter.load(std::memory_order_relaxed)) after(c);
}

struct PlainArgs {
    ID3D11DeviceContext* c;
    UINT count, start;
};
void DrawTail(void* p);

void STDMETHODCALLTYPE D_Draw(ID3D11DeviceContext* c, UINT count, UINT start) {
    hookprof::Scope prof(hookprof::KDraw);
    if (const InsideFn inside = g_drawInside.load(std::memory_order_relaxed); inside && inside()) {
        hookprof::Orig o;
        o_draw(c, count, start);
        return;
    }
    if (mvrpass::SkipDraw(c)) return;  // B11
    srcb::OnDraw(c);  // S2.1
    if (srprobe::Armed()) srprobe::OnDraw(c, _ReturnAddress(), "Draw");  // S2.1 probe frame only
    PlainArgs args{c, count, start};
    // The passes after TAA: the game's sharpen and the windscreen glass (windscreen.h; the rest of the detour as `draw`).
    if (windscreen::OnDraw(c, DrawTail, &args)) return;
    DrawTail(&args);
}

void DrawTail(void* p) {
    const PlainArgs& args = *static_cast<PlainArgs*>(p);
    ID3D11DeviceContext* c = args.c;
    const UINT count = args.count, start = args.start;
    bool taaFix;
    {
        hookprof::SubScope sp(hookprof::STaaDraw);
        taaFix = taaviews::OnDraw(c, false, count);
    }
    const bool mvr = mvrpass::BeforeDraw(c);
    int g = -1;
    Action a = Forward;
    if (Active(c, g)) {
        Protect("Draw", [&] {
            a = OnDraw(c, g, [&] {
                hookprof::Orig o;
                o_draw(c, count, start);
            });
        });
    }
    if (a == Forward) {
        ID3D11ShaderResourceView* keep = nullptr;
        const int bonnet = BonnetBefore(c, &keep);
        {
            hookprof::Orig o;
            o_draw(c, count, start);
        }
        if (bonnet >= 0) BonnetAfter(c, bonnet, keep);
    }
    if (mvr) mvrpass::AfterDraw(c);
    if (taaFix) taaviews::AfterDraw(c);
}

bool HookAll(ID3D11Device* dev, ID3D11DeviceContext* ctx) {
    void** dv = *(void***)dev;
    void** cv = *(void***)ctx;
    void* t[4] = {dv[kSlotCreatePixelShader], cv[kSlotPSSetShader], cv[kSlotDrawIndexed], cv[kSlotDraw]};
    bool ok = hooks::CreateAt(t[0], (void*)D_CreatePixelShader, (void**)&o_createPs, "CreatePixelShader") &&
              hooks::CreateAt(t[1], (void*)D_PSSetShader, (void**)&o_psSet, "PSSetShader") &&
              hooks::CreateAt(t[2], (void*)D_DrawIndexed, (void**)&o_drawIndexed, "DrawIndexed") &&
              hooks::CreateAt(t[3], (void*)D_Draw, (void**)&o_draw, "Draw");
    if (ok) memcpy(g_target, t, sizeof(t));
    return ok;
}

// The game's context and device must call the hooked functions (one d3d11.dll for every device).
bool CheckContext(ID3D11DeviceContext* ctx) {
    ID3D11Device* dev = nullptr;
    ctx->GetDevice(&dev);
    if (!dev) return false;
    void** dv = *(void***)dev;
    void** cv = *(void***)ctx;
    const bool ok = dv[kSlotCreatePixelShader] == g_target[0] && cv[kSlotPSSetShader] == g_target[1] &&
                    cv[kSlotDrawIndexed] == g_target[2] && cv[kSlotDraw] == g_target[3];
    if (!ok)
        Log("Sun effects: the game's D3D11 device does not use the hooked functions (CreatePixelShader %p/%p, "
            "PSSetShader %p/%p, DrawIndexed %p/%p, Draw %p/%p); off for this session",
            dv[kSlotCreatePixelShader], g_target[0], cv[kSlotPSSetShader], g_target[1], cv[kSlotDrawIndexed],
            g_target[2], cv[kSlotDraw], g_target[3]);
    dev->Release();
    return ok;
}

void AddStats(Stats& a, const Stats& b) {
    a.frames += b.frames;
    a.outerFrames += b.outerFrames;
    a.replays += b.replays;
    a.skipped += b.skipped;
    a.composites += b.composites;
    a.anchorComposites += b.anchorComposites;
    a.missed += b.missed;
    a.anchorNoHdr += b.anchorNoHdr;
    a.bonnetOuter += b.bonnetOuter;
    a.bonnetFocus += b.bonnetFocus;
    a.bonnetMissed += b.bonnetMissed;
}

void LogStatsOf(const Stats& s, const char* what) {
    const double f = s.outerFrames ? (double)s.outerFrames : 1;
    Log("Sun effects stats%s: %s; shaders made: rays %llu, flare %llu; frames %llu, with effects in both outer eyes "
        "%llu; per such frame: outer draws repeated %.1f, focus draws skipped %.1f; focus passes at an effect draw "
        "%llu, at the anchor %llu, missed %llu (anchor without the HDR target %llu)",
        what, g_dead ? "STOPPED" : g_enabled ? "on" : "off",
        (unsigned long long)g_created[0].load(), (unsigned long long)g_created[1].load(),
        (unsigned long long)s.frames, (unsigned long long)s.outerFrames, s.replays / f, s.skipped / f,
        (unsigned long long)s.composites, (unsigned long long)s.anchorComposites, (unsigned long long)s.missed,
        (unsigned long long)s.anchorNoHdr);
    Log("Bonnet reflection from outer%s (bonnet_from_outer %s): car paint shaders made %llu; outer's car paint draws "
        "%llu, focus's given outer's image %llu, focus's without it (the game's kept) %llu",
        what, g_bonnetOn ? "on" : "off", (unsigned long long)g_bonnetMade.load(), (unsigned long long)s.bonnetOuter,
        (unsigned long long)s.bonnetFocus, (unsigned long long)s.bonnetMissed);
}

}  // namespace

bool Install() {
    if (g_installed) return true;
    bool ok = false;
    Protect("Install", [&] {
        if (!hooks::Init()) return;
        using CreateFn = HRESULT(WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT, const D3D_FEATURE_LEVEL*, UINT,
                                          UINT, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);
        HMODULE d3d = LoadLibraryW(L"d3d11.dll");
        auto create = d3d ? (CreateFn)GetProcAddress(d3d, "D3D11CreateDevice") : nullptr;
        ID3D11Device* dev = nullptr;
        ID3D11DeviceContext* ctx = nullptr;
        HRESULT hr = create ? create(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                     nullptr, &ctx)
                            : E_NOINTERFACE;
        if (SUCCEEDED(hr) && dev && ctx) ok = HookAll(dev, ctx);
        SafeRelease(ctx);
        SafeRelease(dev);
        Log("Sun effects: D3D11 hooks %s (CreatePixelShader, PSSetShader, DrawIndexed, Draw%s)",
            ok ? "installed" : "FAILED", SUCCEEDED(hr) ? "" : "; no WARP device");
    });
    g_installed = ok;
    return ok;
}

bool SetCreatePsFilter(CreatePsFilterFn create) {
    if (!g_installed) return false;
    g_createPsFilter = create;
    return true;
}

bool SetDrawReplace(DrawReplaceFn replace, InsideFn inside, AfterFn after) {
    if (!g_installed) return false;
    g_drawAfter = after;
    g_drawInside = inside;
    g_drawReplace = replace;
    return true;
}

void OnFrameStart(ID3D11DeviceContext* ctx, bool enabled) {
    Protect("OnFrameStart", [&] {
        const uint32_t gen = gpu::Generation();
        if (gen != g_gen && !g_testing) {
            g_gen = gen;
            ReleaseObjects();
        }
        if (ctx && ctx != g_checkedCtx && g_installed && !g_testing) {
            g_checkedCtx = ctx;
            g_ctxHooked = CheckContext(ctx);
            if (!g_ctxHooked) g_dead = true;
        }
        g_imm.store(ctx, std::memory_order_relaxed);
        ++g_frameNo;
        if (g_valid[0] && g_valid[1]) ++g_st.outerFrames;
        bool on = enabled && g_installed && ctx && !g_dead;
        if (on) {
            ID3D11Device* dev = nullptr;
            ctx->GetDevice(&dev);
            on = dev && EnsurePipeline(dev);
            SafeRelease(dev);
        }
        if (on != g_enabled) Log("Sun effects from outer in the focus views %s", on ? "on" : "off");
        g_enabled = on;
        if (on) ++g_st.frames;
        g_valid[0] = g_valid[1] = false;
    });
}

void BeginGroup(int group, float originX, float originY, float scaleX, float scaleY) {
    Protect("BeginGroup", [&] {
        if (group < 2) {
            g_valid[group] = false;
            g_acc[group].checkedFor = nullptr;
            g_afterSun = false;
        } else {
            g_done = false;
            g_origin[0] = originX;
            g_origin[1] = originY;
            g_scale[0] = scaleX;
            g_scale[1] = scaleY;
        }
        g_group.store(group, std::memory_order_relaxed);
    });
}

void EndGroup() {
    const int g = g_group.exchange(-1, std::memory_order_relaxed);
    Protect("EndGroup", [&] {
        if (g < 2) {
            g_afterSun = false;
            return;
        }
        if (g_enabled && g_valid[g - 2] && !g_done) {
            ++g_st.missed;
            if (!g_missLogged) {
                g_missLogged = true;
                Log("Sun effects: focus %s without the pass although outer drew effects (counted from now on)",
                    g == 2 ? "L" : "R");
            }
        }
    });
}

void SetBonnetFromOuter(bool on) {
    if (on == g_bonnetOn) return;
    g_bonnetOn = on;
    Log("Sun effects: bonnet reflection of the focus views from outer %s", on ? "on" : "off");
}

void OnSetTargets(ID3D11DeviceContext* c, UINT n, ID3D11RenderTargetView* const* rtv) {
    if (c != g_imm.load(std::memory_order_relaxed)) return;
    g_curRt0 = n && rtv ? rtv[0] : nullptr;
}

std::string Describe() {
    char buf[160];
    snprintf(buf, sizeof(buf), "sun effects from outer %s (shaders made: rays %llu, flare %llu)",
             g_dead ? "STOPPED" : g_enabled ? "on" : "off", (unsigned long long)g_created[0].load(),
             (unsigned long long)g_created[1].load());
    return buf;
}

void LogStats(bool totals) {
    if (!g_installed) return;
    AddStats(g_total, g_st);
    LogStatsOf(g_st, "");
    g_st = Stats{};
    if (totals) LogStatsOf(g_total, " (session)");
}

void Shutdown() {
    Protect("Shutdown", [&] {
        g_group = -1;
        g_enabled = false;
        g_imm = nullptr;
        g_checkedCtx = nullptr;
        g_ctxHooked = false;
        ReleaseObjects();
    });
}

void NewPsEpoch() { ++g_psEpoch; }
ID3D11PixelShader* CurrentPs(ID3D11DeviceContext* c) {
    // The hook sees the game's calls only when its device uses the hooked functions (CheckContext). A fault that stops
    // the sun effects (g_dead) does not stop the hook.
    const bool tracked = g_installed && g_ctxHooked && c == g_imm.load(std::memory_order_relaxed);
    if (tracked && g_curPsEpoch == g_psEpoch) return g_curPs;
    ID3D11PixelShader* ps = nullptr;
    c->PSGetShader(&ps, nullptr, nullptr);
    if (ps) ps->Release();  // the context keeps it bound
    ++g_psReads;
    hookprof::Count(hookprof::CPsGet);
    if (tracked) {
        g_curPs = ps;
        g_curKind = Lookup(ps, &g_curArg);
        g_curPsEpoch = g_psEpoch;
    }
    return ps;
}
uint64_t PsReads() { return g_psReads; }

bool TestInstall(ID3D11Device* dev, ID3D11DeviceContext* ctx) {
    g_testing = true;
    if (!g_installed) g_installed = hooks::Init() && HookAll(dev, ctx);
    g_imm.store(ctx, std::memory_order_relaxed);  // the PSSetShader hook follows the test context (CurrentPs)
    g_ctxHooked = g_installed;
    return g_installed;
}

void TestRegister(ID3D11PixelShader* ps, Kind kind) { SetKind(ps, kind); }

TestState TestGet() {
    TestState t;
    Stats s = g_total;
    AddStats(s, g_st);
    t.replays = s.replays;
    t.skipped = s.skipped;
    t.composites = s.composites;
    t.anchorComposites = s.anchorComposites;
    t.missed = s.missed;
    t.faults = g_dead ? 1 : 0;
    t.bonnetMade = g_bonnetMade.load();
    t.bonnetOuter = s.bonnetOuter;
    t.bonnetFocus = s.bonnetFocus;
    t.bonnetMissed = s.bonnetMissed;
    return t;
}

}  // namespace qvfr::sunfx
