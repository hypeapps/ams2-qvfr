#include "compose.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "common.h"
#include "compose_ps.h"  // kComposePS (fxc from shaders/compose.hlsl)
#include "compose_vs.h"  // kComposeVS

namespace qvfr::compose {

namespace {

// Matches cbuffer Params in compose.hlsl.
struct Constants {
    float gridW, gridH, x0, y0, focusW, focusH, margin, feather;
};
static_assert(sizeof(Constants) == 32);

struct Pipeline {
    ID3D11Device* dev = nullptr;  // referenced: the objects below belong to it
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11SamplerState* sampler = nullptr;
    ID3D11Buffer* cb = nullptr;
    bool failed = false;
};

struct Output {
    ID3D11Texture2D* tex = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    D3D11_TEXTURE2D_DESC desc{};
    bool logged = false;
    bool failed = false;  // creation failed for desc: not retried every Submit
};

// Shader resource views of the source textures. The view holds a reference, so a cached texture address cannot be
// reused by a new texture while its entry lives.
struct CachedSrv {
    ID3D11Texture2D* tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};
constexpr int kSrvCache = 8;  // the game alternates two Submit textures per eye, plus the proxy's copies

// The pipeline state Draw changes, saved before and restored after it.
struct State {
    static constexpr UINT kMaxCi = 256;
    static constexpr UINT kMaxVp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    ID3D11InputLayout* il = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY topo = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ID3D11VertexShader* vs = nullptr;
    ID3D11HullShader* hs = nullptr;
    ID3D11DomainShader* ds = nullptr;
    ID3D11GeometryShader* gs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11ClassInstance* ci[5][kMaxCi] = {};
    UINT nci[5] = {};
    ID3D11ShaderResourceView* srv[2] = {};
    ID3D11SamplerState* sampler = nullptr;
    ID3D11Buffer* cb = nullptr;
    ID3D11RasterizerState* rs = nullptr;
    D3D11_VIEWPORT vp[kMaxVp] = {};
    UINT nvp = 0;
    D3D11_RECT sc[kMaxVp] = {};
    UINT nsc = 0;
    ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    ID3D11BlendState* bs = nullptr;
    float blendFactor[4] = {};
    UINT sampleMask = 0;
    ID3D11DepthStencilState* dss = nullptr;
    UINT stencilRef = 0;
};

Pipeline g_p;
Output g_out[2];
CachedSrv g_srv[kSrvCache];
int g_srvNext = 0;
ID3D11Texture2D* g_scratch[2][2] = {};  // [eye][outer, focus]: copies of sources without shader resource binding
bool g_scratchFailed[2][2] = {};
State g_state;
// Failure sites, each logged once per eye until its message changes (the eyes and the sources alternate).
enum Site {
    SitePipeline,
    SiteSize,
    SiteFormat,
    SiteWindow,
    SiteOutput,
    SiteCopy,
    SiteView = SiteCopy + 2,
    SiteMap = SiteView + 2,
    kSites
};
char g_lastError[2][kSites][256] = {};

template <class T>
void SafeRelease(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// Logs a failure once per eye and site until a different one happens there (the caller counts the eyes sent without
// the composition).
void Fail(int eye, int site, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    char* last = g_lastError[eye][site];
    if (strcmp(buf, last) == 0) return;
    snprintf(last, sizeof(g_lastError[eye][site]), "%s", buf);
    Log("Compose: %s; this eye goes out without the composition", buf);
}

// _SRGB views for the 8-bit formats, so the compositor's sRGB reading and our filtering agree.
DXGI_FORMAT ViewFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default: return f;
    }
}

void ReleaseOutputs() {
    for (Output& o : g_out) {
        SafeRelease(o.rtv);
        SafeRelease(o.tex);
        o = Output{};
    }
    for (CachedSrv& c : g_srv) {
        SafeRelease(c.srv);
        c.tex = nullptr;
    }
    g_srvNext = 0;
    for (auto& eye : g_scratch)
        for (ID3D11Texture2D*& t : eye) SafeRelease(t);
    for (auto& eye : g_scratchFailed)
        for (bool& f : eye) f = false;
}

bool EnsurePipeline(ID3D11Device* dev) {
    if (g_p.dev != dev) {
        Release();
        g_p.dev = dev;
        dev->AddRef();
    }
    if (g_p.failed) return false;
    if (g_p.vs) return true;
    HRESULT hr = dev->CreateVertexShader(kComposeVS, sizeof(kComposeVS), nullptr, &g_p.vs);
    if (SUCCEEDED(hr)) hr = dev->CreatePixelShader(kComposePS, sizeof(kComposePS), nullptr, &g_p.ps);
    if (SUCCEEDED(hr)) {
        D3D11_SAMPLER_DESC s{};
        s.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        s.AddressU = s.AddressV = s.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        s.ComparisonFunc = D3D11_COMPARISON_NEVER;
        s.MaxLOD = D3D11_FLOAT32_MAX;
        hr = dev->CreateSamplerState(&s, &g_p.sampler);
    }
    if (SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC b{};
        b.ByteWidth = sizeof(Constants);
        b.Usage = D3D11_USAGE_DYNAMIC;
        b.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        b.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = dev->CreateBuffer(&b, nullptr, &g_p.cb);
    }
    if (FAILED(hr)) {
        SafeRelease(g_p.vs);
        SafeRelease(g_p.ps);
        SafeRelease(g_p.sampler);
        SafeRelease(g_p.cb);
        g_p.failed = true;
        Fail(0, SitePipeline, "cannot create the shaders and states (0x%08lX, device feature level 0x%X)",
             (unsigned long)hr, (unsigned)dev->GetFeatureLevel());
        return false;
    }
    Log("Compose: shaders ready (vs %zu B, ps %zu B, feature level 0x%X)", sizeof(kComposeVS), sizeof(kComposePS),
        (unsigned)dev->GetFeatureLevel());
    return true;
}

bool EnsureOutput(int eye, const Layout& l, DXGI_FORMAT fmt) {
    Output& o = g_out[eye];
    if (o.tex && (o.desc.Width != l.gridW || o.desc.Height != l.gridH || o.desc.Format != fmt)) {
        SafeRelease(o.rtv);
        SafeRelease(o.tex);
        o = Output{};
    }
    if (o.tex) return true;
    if (o.failed && o.desc.Width == l.gridW && o.desc.Height == l.gridH && o.desc.Format == fmt) return false;
    D3D11_TEXTURE2D_DESC d{};
    d.Width = l.gridW;
    d.Height = l.gridH;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = fmt;
    d.SampleDesc = {1, 0};
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    HRESULT hr = g_p.dev->CreateTexture2D(&d, nullptr, &o.tex);
    if (SUCCEEDED(hr)) {
        D3D11_RENDER_TARGET_VIEW_DESC v{};
        v.Format = ViewFormat(fmt);
        v.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        hr = g_p.dev->CreateRenderTargetView(o.tex, &v, &o.rtv);
    }
    if (FAILED(hr)) {
        SafeRelease(o.tex);
        o.desc = d;
        o.failed = true;
        Fail(eye, SiteOutput, "cannot create the %s eye output %u x %u fmt %d (0x%08lX)", eye ? "right" : "left",
             d.Width, d.Height, (int)fmt, (unsigned long)hr);
        return false;
    }
    o.desc = d;
    o.failed = false;
    Log("Compose: %s eye output %p, %u x %u fmt %d (views fmt %d)", eye ? "right" : "left", (void*)o.tex, d.Width,
        d.Height, (int)fmt, (int)ViewFormat(fmt));
    return true;
}

// A view of the source; a texture the game made without shader resource binding is copied first.
ID3D11ShaderResourceView* SourceView(ID3D11DeviceContext* ctx, int eye, int which, ID3D11Texture2D* tex,
                                     const D3D11_TEXTURE2D_DESC& d) {
    if (!(d.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
        ID3D11Texture2D*& s = g_scratch[eye][which];
        D3D11_TEXTURE2D_DESC have{};
        if (s) s->GetDesc(&have);
        if (s && (have.Width != d.Width || have.Height != d.Height || have.Format != d.Format ||
                  have.MipLevels != d.MipLevels)) {
            SafeRelease(s);
        }
        if (!s && g_scratchFailed[eye][which]) return nullptr;
        if (!s) {
            D3D11_TEXTURE2D_DESC n = d;
            n.Usage = D3D11_USAGE_DEFAULT;
            n.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            n.CPUAccessFlags = 0;
            n.MiscFlags = 0;
            HRESULT hr = g_p.dev->CreateTexture2D(&n, nullptr, &s);
            if (FAILED(hr)) {
                s = nullptr;
                g_scratchFailed[eye][which] = true;
                Fail(eye, SiteCopy + which, "cannot create a copy of the %s source %u x %u fmt %d (0x%08lX)",
                     which ? "focus" : "outer", d.Width, d.Height, (int)d.Format, (unsigned long)hr);
                return nullptr;
            }
            Log("Compose: %s eye %s source %p has no shader resource binding; copied into %p every frame",
                eye ? "right" : "left", which ? "focus" : "outer", (void*)tex, (void*)s);
        }
        ctx->CopyResource(s, tex);
        tex = s;
    }
    for (const CachedSrv& c : g_srv)
        if (c.tex == tex) return c.srv;
    D3D11_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = ViewFormat(d.Format);
    v.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    v.Texture2D.MipLevels = 1;
    ID3D11ShaderResourceView* srv = nullptr;
    HRESULT hr = g_p.dev->CreateShaderResourceView(tex, &v, &srv);
    if (FAILED(hr) || !srv) {
        Fail(eye, SiteView + which, "cannot create a view of the %s source fmt %d as fmt %d (0x%08lX)",
             which ? "focus" : "outer", (int)d.Format, (int)v.Format, (unsigned long)hr);
        return nullptr;
    }
    CachedSrv& c = g_srv[g_srvNext];
    g_srvNext = (g_srvNext + 1) % kSrvCache;
    SafeRelease(c.srv);
    c.tex = tex;
    c.srv = srv;
    return srv;
}

void Save(ID3D11DeviceContext* c, State& s) {
    c->IAGetInputLayout(&s.il);
    c->IAGetPrimitiveTopology(&s.topo);
    for (UINT& n : s.nci) n = State::kMaxCi;
    c->VSGetShader(&s.vs, s.ci[0], &s.nci[0]);
    c->HSGetShader(&s.hs, s.ci[1], &s.nci[1]);
    c->DSGetShader(&s.ds, s.ci[2], &s.nci[2]);
    c->GSGetShader(&s.gs, s.ci[3], &s.nci[3]);
    c->PSGetShader(&s.ps, s.ci[4], &s.nci[4]);
    c->PSGetShaderResources(0, 2, s.srv);
    c->PSGetSamplers(0, 1, &s.sampler);
    c->PSGetConstantBuffers(0, 1, &s.cb);
    c->RSGetState(&s.rs);
    s.nvp = State::kMaxVp;
    c->RSGetViewports(&s.nvp, s.vp);
    s.nsc = State::kMaxVp;
    c->RSGetScissorRects(&s.nsc, s.sc);
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, s.rtv, &s.dsv);
    c->OMGetBlendState(&s.bs, s.blendFactor, &s.sampleMask);
    c->OMGetDepthStencilState(&s.dss, &s.stencilRef);
}

// Puts the saved state back and drops the references the getters took. The shader resources go back before the
// render targets, so a game texture that is both is never bound twice by our doing.
void Restore(ID3D11DeviceContext* c, State& s) {
    c->IASetInputLayout(s.il);
    c->IASetPrimitiveTopology(s.topo);
    c->VSSetShader(s.vs, s.ci[0], s.nci[0]);
    c->HSSetShader(s.hs, s.ci[1], s.nci[1]);
    c->DSSetShader(s.ds, s.ci[2], s.nci[2]);
    c->GSSetShader(s.gs, s.ci[3], s.nci[3]);
    c->PSSetShader(s.ps, s.ci[4], s.nci[4]);
    c->PSSetShaderResources(0, 2, s.srv);
    c->PSSetSamplers(0, 1, &s.sampler);
    c->PSSetConstantBuffers(0, 1, &s.cb);
    c->RSSetState(s.rs);
    c->RSSetViewports(s.nvp, s.vp);
    c->RSSetScissorRects(s.nsc, s.sc);
    c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, s.rtv, s.dsv);
    c->OMSetBlendState(s.bs, s.blendFactor, s.sampleMask);
    c->OMSetDepthStencilState(s.dss, s.stencilRef);
    SafeRelease(s.il);
    SafeRelease(s.vs);
    SafeRelease(s.hs);
    SafeRelease(s.ds);
    SafeRelease(s.gs);
    SafeRelease(s.ps);
    for (int k = 0; k < 5; ++k) {
        for (UINT i = 0; i < s.nci[k] && i < State::kMaxCi; ++i) SafeRelease(s.ci[k][i]);
        s.nci[k] = 0;
    }
    for (auto*& v : s.srv) SafeRelease(v);
    SafeRelease(s.sampler);
    SafeRelease(s.cb);
    SafeRelease(s.rs);
    for (auto*& v : s.rtv) SafeRelease(v);
    SafeRelease(s.dsv);
    SafeRelease(s.bs);
    SafeRelease(s.dss);
}

}  // namespace

ID3D11Texture2D* Draw(ID3D11DeviceContext* ctx, int eye, ID3D11Texture2D* outer, ID3D11Texture2D* focus,
                      const Layout& l) {
    if (!ctx || !outer || !focus || eye < 0 || eye > 1) return nullptr;
    D3D11_TEXTURE2D_DESC od{}, fd{};
    outer->GetDesc(&od);
    focus->GetDesc(&fd);
    const char* side = eye ? "right" : "left";
    if (od.Width != l.focusW || od.Height != l.focusH || fd.Width != l.focusW || fd.Height != l.focusH) {
        Fail(eye, SiteSize, "%s eye sources outer %u x %u, focus %u x %u are not the focus size %u x %u", side,
             od.Width, od.Height, fd.Width, fd.Height, l.focusW, l.focusH);
        return nullptr;
    }
    if (od.SampleDesc.Count != 1 || fd.SampleDesc.Count != 1 || od.Format != fd.Format) {
        Fail(eye, SiteFormat,
             "%s eye sources: samples %u and %u, formats %d and %d (expected one sample, the same format)", side,
             od.SampleDesc.Count, fd.SampleDesc.Count, (int)od.Format, (int)fd.Format);
        return nullptr;
    }
    if (!l.gridW || !l.gridH || l.x0 + l.focusW > l.gridW || l.y0 + l.focusH > l.gridH) {
        Fail(eye, SiteWindow, "focus window (%u, %u) %u x %u does not fit the grid %u x %u", l.x0, l.y0, l.focusW,
             l.focusH, l.gridW, l.gridH);
        return nullptr;
    }
    ID3D11Device* dev = nullptr;
    ctx->GetDevice(&dev);
    if (!dev) return nullptr;
    bool ready = EnsurePipeline(dev);
    dev->Release();  // the pipeline holds its own reference
    if (!ready || !EnsureOutput(eye, l, od.Format)) return nullptr;
    ID3D11ShaderResourceView* srv[2] = {SourceView(ctx, eye, 0, outer, od), SourceView(ctx, eye, 1, focus, fd)};
    if (!srv[0] || !srv[1]) return nullptr;
    D3D11_MAPPED_SUBRESOURCE m{};
    HRESULT hr = ctx->Map(g_p.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    if (FAILED(hr)) {
        Fail(eye, SiteMap, "cannot map the constant buffer (0x%08lX)", (unsigned long)hr);
        return nullptr;
    }
    const Constants k = {(float)l.gridW,  (float)l.gridH,  (float)l.x0, (float)l.y0,
                         (float)l.focusW, (float)l.focusH, l.marginPx,  l.featherPx > 0 ? l.featherPx : 1e-3f};
    memcpy(m.pData, &k, sizeof(k));
    ctx->Unmap(g_p.cb, 0);

    Output& o = g_out[eye];
    Save(ctx, g_state);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(g_p.vs, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->PSSetShader(g_p.ps, nullptr, 0);
    ctx->OMSetRenderTargets(1, &o.rtv, nullptr);  // first, so no source stays bound as a render target
    ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(nullptr, 0);
    ctx->RSSetState(nullptr);
    const D3D11_VIEWPORT vp = {0, 0, (float)l.gridW, (float)l.gridH, 0, 1};
    ctx->RSSetViewports(1, &vp);
    ctx->PSSetShaderResources(0, 2, srv);
    ctx->PSSetSamplers(0, 1, &g_p.sampler);
    ctx->PSSetConstantBuffers(0, 1, &g_p.cb);
    ctx->Draw(3, 0);
    Restore(ctx, g_state);
    if (!o.logged) {
        o.logged = true;
        Log("Compose: first %s eye: outer %p + focus %p (%u x %u fmt %d, shader resource binding %s/%s) -> %p, focus at "
            "(%u, %u), margin %.0f px, feather %.0f px",
            side, (void*)outer, (void*)focus, od.Width, od.Height, (int)od.Format,
            (od.BindFlags & D3D11_BIND_SHADER_RESOURCE) ? "yes" : "no",
            (fd.BindFlags & D3D11_BIND_SHADER_RESOURCE) ? "yes" : "no", (void*)o.tex, l.x0, l.y0, l.marginPx,
            l.featherPx);
    }
    return o.tex;
}

void Release() {
    ReleaseOutputs();
    SafeRelease(g_p.vs);
    SafeRelease(g_p.ps);
    SafeRelease(g_p.sampler);
    SafeRelease(g_p.cb);
    SafeRelease(g_p.dev);
    g_p = Pipeline{};
    memset(g_lastError, 0, sizeof(g_lastError));
}

}  // namespace qvfr::compose
