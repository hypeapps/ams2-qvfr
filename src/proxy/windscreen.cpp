#include "windscreen.h"

#include <windows.h>

#include <d3d11_1.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "clones.h"
#include "common.h"
#include "windscreen_dilate_cs.h"  // kWindscreenDilateCS (fxc from shaders/windscreen.hlsl)

namespace qvfr::windscreen {
namespace {

constexpr int kMaxLayerPs = 1024;
constexpr int kScratch = 4;  // outer and focus sizes, with room

std::atomic<ID3D11PixelShader*> g_sharpPs[4] = {};
std::atomic<ID3D11PixelShader*> g_layerPs[kMaxLayerPs] = {};
uint32_t g_layerSum[kMaxLayerPs][4] = {};  // DXBC checksum of each (the names in x64\qvfr\ps_dump)
// The vehicle glass among them (rainControlMap_tex: the glass, sun strip and frit). Only these test the glass's depth:
// the other shaders with the shared constants are any transparent object after TAA (other cars' windows, trackside
// glass), which would draw over the outline of whatever stands in front of them.
bool g_layerGlass[kMaxLayerPs] = {};
std::atomic<int> g_layerCount{0}, g_glassCount{0};
std::atomic<bool> g_dead{false};
std::atomic<bool> g_layersOn{true};
std::atomic<int> g_depthSwitch{-1};  // the quad switch "windscreen_depth" over the setting; -1 = the setting

// Render thread, immediate context.
bool g_curSharp = false;              // the bound pixel shader is the sharpen
bool g_curLayer = false;              // the bound pixel shader is a layer shader (looked up after the sharpen only)
bool g_curGlass = false;              // ... and a vehicle glass shader
ID3D11PixelShader* g_curPs = nullptr;  // the bound pixel shader (compared only)
bool g_afterSharpen = false;          // the group's sharpen drew
ID3D11Resource* g_sharpTarget = nullptr;  // its render target (compared only, no reference)

std::atomic<uint64_t> g_sharpCopied{0}, g_sharpBlended{0}, g_sharpKept{0}, g_layerSkipped{0}, g_depthBuilds{0},
    g_depthFails{0}, g_depthDraws{0}, g_depthOther{0};
bool g_scaleLogged[2] = {};  // the first glass's depth of each view kind with DLSS SR
int g_logs = 0;

// The glass's depth (shaders/windscreen.hlsl CSDilate): a copy of the scene's depth (D32, no stencil), its depth the
// farthest within the radius; made once per group at the first glass draw. A compute pass, so none of the game's draw
// state changes: the copy in `src`, the result in `mid` (unordered access), copied into `dst` (the depth buffer).
struct DepthScratch {
    D3D11_TEXTURE2D_DESC desc{};              // the game's texture
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};  // the game's view
    ID3D11Texture2D *src = nullptr, *mid = nullptr, *dst = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
};
DepthScratch g_dep[kScratch];
int g_depNext = 0;
ID3D11ComputeShader* g_dilCs = nullptr;
ID3D11Buffer* g_dcb = nullptr;
bool g_depFailed = false;
ID3D11DepthStencilView* g_dilDsv = nullptr;  // this group's (owned by a scratch)
ID3D11DepthStencilView* g_dilFor = nullptr;  // the game's view it was made from (compared only)
bool g_dilTried = false;                     // this group: made or failed once

// The sharpen at a fraction of its strength: drawn over the TAA result with this blend.
ID3D11BlendState* g_sharpBlend = nullptr;
bool g_sharpBlendFailed = false;
bool g_sharpStateLogged = false;
bool g_sharpOwnBlend = false;  // the game draws its sharpen with blending: only on or off then

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

void Fail(const char* where, unsigned long code) {
    if (g_dead.exchange(true)) return;
    Log("Windscreen: ERROR: exception 0x%08lX in %s; off for this session (the game's passes after TAA as they are)",
        code, where);
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

bool IsSharp(ID3D11PixelShader* ps) {
    if (!ps) return false;
    for (auto& s : g_sharpPs)
        if (s.load(std::memory_order_relaxed) == ps) return true;
    return false;
}

// The layer shader's slot, or -1.
int LayerIndex(ID3D11PixelShader* ps) {
    if (!ps) return -1;
    for (int i = 0; i < kMaxLayerPs; ++i)
        if (g_layerPs[i].load(std::memory_order_relaxed) == ps) return i;
    return -1;
}

bool Contains(const void* code, size_t len, const char* text) {
    const uint8_t* b = (const uint8_t*)code;
    const size_t n = strlen(text);
    return std::search(b, b + len, text, text + n) != b + len;
}

// The sharpen's strength as a fraction of the game's (0..1).
float SharpenAmount() { return std::clamp(Cfg().gameSharpen, 0.0f, 1.0f); }

int DepthRadius() {
    const int s = g_depthSwitch.load(std::memory_order_relaxed);
    return std::clamp(s >= 0 ? s : Cfg().windscreenDepth, 0, 3);
}

// The radius in the depth texture's pixels: the setting counts render pixels. With DLSS SR the scene's depth reaches
// the passes after TAA upscaled from the render size by nearest (a render pixel is a block of view pixels), while
// DLSS put the outline anywhere within the block.
int OutputRadius(int radius) {
    clones::SrGroup g;
    if (!clones::SrGroupNow(g) || !g.post || g.rw == 0 || g.ow <= g.rw) return radius;
    const float scale = (float)g.ow / (float)g.rw;
    const int r = std::clamp((int)std::lround(radius * scale), radius, 8);
    const int k = g.kind == clones::Outer ? 0 : 1;
    if (!g_scaleLogged[k]) {
        g_scaleLogged[k] = true;
        Log("Windscreen: glass's depth in %s with DLSS SR: render %ux%u, view %ux%u, radius %d render px = %d px",
            k == 0 ? "outer" : "focus", g.rw, g.rh, g.ow, g.oh, radius, r);
    }
    return r;
}

void ReleaseDepth(DepthScratch& s) {
    Rel(s.srv);
    Rel(s.uav);
    Rel(s.dsv);
    Rel(s.src);
    Rel(s.mid);
    Rel(s.dst);
    s = DepthScratch{};
}

bool EnsureDepthShader(ID3D11Device* dev) {
    if (g_dilCs && g_dcb) return true;
    if (g_depFailed) return false;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 32;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(dev->CreateComputeShader(kWindscreenDilateCS, sizeof(kWindscreenDilateCS), nullptr, &g_dilCs)) ||
        FAILED(dev->CreateBuffer(&bd, nullptr, &g_dcb))) {
        Rel(g_dilCs);
        Rel(g_dcb);
        g_depFailed = true;
        Log("Windscreen: the glass's depth shader could not be made; the glass tests the scene's depth as it is");
        return false;
    }
    return true;
}

void NoteDepth(const char* what, const D3D11_TEXTURE2D_DESC& d) {
    if (g_logs >= 12) return;
    ++g_logs;
    Log("Windscreen: glass's depth not made: %s (depth %ux%u x%u format %d, %u samples)", what, d.Width, d.Height,
        d.ArraySize, (int)d.Format, d.SampleDesc.Count);
}

// The group's first glass draw: the glass's depth, the scene's made farther within the radius. Null: none (the game's).
ID3D11DepthStencilView* MakeGlassDepth(ID3D11DeviceContext* c, ID3D11DepthStencilView* gameDsv, const D3D11_VIEWPORT& vp,
                                       int radius) {
    ID3D11Resource* res = nullptr;
    gameDsv->GetResource(&res);
    ID3D11Texture2D* tex = nullptr;
    if (res) res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex);
    Rel(res);
    D3D11_TEXTURE2D_DESC td{};
    if (tex) tex->GetDesc(&td);
    D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
    gameDsv->GetDesc(&dd);
    const char* why = nullptr;
    UINT slice = 0, mip = 0;
    if (dd.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2D) {
        mip = dd.Texture2D.MipSlice;
    } else if (dd.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2DARRAY) {
        mip = dd.Texture2DArray.MipSlice;
        slice = dd.Texture2DArray.FirstArraySlice;
    } else {
        why = "the view is not of a 2D texture";
    }
    if (!why && !tex) why = "no texture";
    if (!why && td.SampleDesc.Count != 1) why = "multisampled";
    // D32 only: a compute shader cannot write a format with stencil.
    if (!why && td.Format != DXGI_FORMAT_R32_TYPELESS && td.Format != DXGI_FORMAT_D32_FLOAT) why = "format not handled";
    if (!why && mip != 0) why = "not mip 0";
    ID3D11Device* dev = nullptr;
    c->GetDevice(&dev);
    if (!why && (!dev || !EnsureDepthShader(dev))) why = "no shader";
    DepthScratch* s = nullptr;
    if (!why) {
        dd.Flags = 0;  // ours is written
        for (DepthScratch& d : g_dep)
            if (d.dsv && d.desc.Width == td.Width && d.desc.Height == td.Height && d.desc.ArraySize == td.ArraySize &&
                d.desc.MipLevels == td.MipLevels && d.desc.Format == td.Format &&
                memcmp(&d.dsvDesc, &dd, sizeof(dd)) == 0) {
                s = &d;
                break;
            }
        if (!s) {
            s = &g_dep[g_depNext];
            g_depNext = (g_depNext + 1) % kScratch;
            ReleaseDepth(*s);
            D3D11_TEXTURE2D_DESC d = td;
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.Usage = D3D11_USAGE_DEFAULT;
            d.CPUAccessFlags = 0;
            d.MiscFlags = 0;
            d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            D3D11_TEXTURE2D_DESC m = d;
            m.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
            D3D11_TEXTURE2D_DESC w = d;
            w.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
            sv.Format = DXGI_FORMAT_R32_FLOAT;
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
            sv.Texture2DArray.MipLevels = 1;
            sv.Texture2DArray.ArraySize = td.ArraySize;
            D3D11_UNORDERED_ACCESS_VIEW_DESC uv{};
            uv.Format = DXGI_FORMAT_R32_FLOAT;
            uv.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
            uv.Texture2DArray.ArraySize = td.ArraySize;
            const bool ok = SUCCEEDED(dev->CreateTexture2D(&d, nullptr, &s->src)) &&
                            SUCCEEDED(dev->CreateTexture2D(&m, nullptr, &s->mid)) &&
                            SUCCEEDED(dev->CreateTexture2D(&w, nullptr, &s->dst)) &&
                            SUCCEEDED(dev->CreateShaderResourceView(s->src, &sv, &s->srv)) &&
                            SUCCEEDED(dev->CreateUnorderedAccessView(s->mid, &uv, &s->uav)) &&
                            SUCCEEDED(dev->CreateDepthStencilView(s->dst, &dd, &s->dsv));
            if (!ok) {
                ReleaseDepth(*s);
                s = nullptr;
                why = "textures not made";
            } else {
                s->desc = td;
                s->dsvDesc = dd;
                Log("Windscreen: glass's depth textures %ux%u x%u format %d (view format %d, slice %u)", td.Width,
                    td.Height, td.ArraySize, (int)td.Format, (int)dd.Format, slice);
            }
        }
    }
    ID3D11DepthStencilView* out = nullptr;
    if (!why) {
        c->CopyResource(s->src, tex);
        if (td.ArraySize > 1) c->CopyResource(s->mid, tex);  // the other slices as they are
        const UINT l = (UINT)std::max(0.0f, vp.TopLeftX), t = (UINT)std::max(0.0f, vp.TopLeftY);
        const UINT r = std::min(td.Width, (UINT)(vp.TopLeftX + vp.Width));
        const UINT b = std::min(td.Height, (UINT)(vp.TopLeftY + vp.Height));
        struct {
            uint32_t origin[2], size[2], tex[2], slice;
            int32_t radius;
        } cb{{l, t}, {r > l ? r - l : 1, b > t ? b - t : 1}, {td.Width, td.Height}, slice, OutputRadius(radius)};
        static_assert(sizeof(cb) == 32, "DepthParams in shaders/windscreen.hlsl");
        // Compute slot 0 only, the game's put back as it was.
        ID3D11ComputeShader* oCs = nullptr;
        ID3D11Buffer* oCb = nullptr;
        ID3D11ShaderResourceView* oSrv = nullptr;
        ID3D11UnorderedAccessView* oUav = nullptr;
        c->CSGetShader(&oCs, nullptr, nullptr);
        c->CSGetConstantBuffers(0, 1, &oCb);
        c->CSGetShaderResources(0, 1, &oSrv);
        c->CSGetUnorderedAccessViews(0, 1, &oUav);
        c->UpdateSubresource(g_dcb, 0, nullptr, &cb, 0, 0);
        c->CSSetShader(g_dilCs, nullptr, 0);
        c->CSSetConstantBuffers(0, 1, &g_dcb);
        c->CSSetShaderResources(0, 1, &s->srv);
        c->CSSetUnorderedAccessViews(0, 1, &s->uav, nullptr);
        c->Dispatch((td.Width + 7) / 8, (td.Height + 7) / 8, 1);
        ID3D11ShaderResourceView* noSrv = nullptr;
        ID3D11UnorderedAccessView* noUav = nullptr;
        c->CSSetShaderResources(0, 1, &noSrv);
        c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
        c->CSSetShader(oCs, nullptr, 0);
        c->CSSetConstantBuffers(0, 1, &oCb);
        c->CSSetShaderResources(0, 1, &oSrv);
        c->CSSetUnorderedAccessViews(0, 1, &oUav, nullptr);
        Rel(oCs);
        Rel(oCb);
        Rel(oSrv);
        Rel(oUav);
        c->CopyResource(s->dst, s->mid);
        out = s->dsv;
        if (g_depthBuilds.fetch_add(1, std::memory_order_relaxed) == 0)
            Log("Windscreen: first glass's depth made (%ux%u at %u,%u, radius %d px)", cb.size[0], cb.size[1], l, t,
                cb.radius);
    } else {
        g_depthFails.fetch_add(1, std::memory_order_relaxed);
        NoteDepth(why, td);
    }
    Rel(dev);
    Rel(tex);
    return out;
}

// The TAA result (t0) copied into the sharpen's target in the viewport, when both are 2D textures of one size and
// format. False: not copied.
bool CopySharpenSource(ID3D11DeviceContext* c) {
    bool done = false;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    c->PSGetShaderResources(0, 1, &srv);
    c->OMGetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp{};
    UINT nvp = 1;
    c->RSGetViewports(&nvp, &vp);
    ID3D11Resource *src = nullptr, *dst = nullptr;
    if (srv) srv->GetResource(&src);
    if (rtv) rtv->GetResource(&dst);
    ID3D11Texture2D *st = nullptr, *dt = nullptr;
    if (src) src->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&st);
    if (dst) dst->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&dt);
    D3D11_TEXTURE2D_DESC sd{}, dd{};
    if (st) st->GetDesc(&sd);
    if (dt) dt->GetDesc(&dd);
    if (st && dt && nvp && sd.Width == dd.Width && sd.Height == dd.Height && sd.Format == dd.Format &&
        sd.SampleDesc.Count == 1 && dd.SampleDesc.Count == 1) {
        D3D11_SHADER_RESOURCE_VIEW_DESC svd{};
        D3D11_RENDER_TARGET_VIEW_DESC rvd{};
        srv->GetDesc(&svd);
        rtv->GetDesc(&rvd);
        const UINT ss = svd.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2DARRAY ? svd.Texture2DArray.FirstArraySlice : 0;
        const UINT ds = rvd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY ? rvd.Texture2DArray.FirstArraySlice : 0;
        const UINT l = (UINT)std::max(0.0f, vp.TopLeftX), t = (UINT)std::max(0.0f, vp.TopLeftY);
        const UINT r = std::min(dd.Width, (UINT)(vp.TopLeftX + vp.Width));
        const UINT b = std::min(dd.Height, (UINT)(vp.TopLeftY + vp.Height));
        if (r > l && b > t) {
            const D3D11_BOX box{l, t, 0, r, b, 1};
            c->CopySubresourceRegion(dt, D3D11CalcSubresource(0, ds, dd.MipLevels), l, t, 0, st,
                                     D3D11CalcSubresource(0, ss, sd.MipLevels), &box);
            done = true;
        }
    }
    Rel(st);
    Rel(dt);
    Rel(src);
    Rel(dst);
    Rel(srv);
    Rel(rtv);
    return done;
}

// The game's sharpen draw at `amount` of its strength (0..1). The luma sharpen adds strength x the pixel's luma
// detail (its clamp, sharp_clamp 100, is never reached), so the TAA result plus `amount` of the sharpen's change is
// the sharpen at that strength: the result is copied into the target and the game's draw blended over it with
// that factor. True: handled here.
bool Sharpen(ID3D11DeviceContext* c, float amount, DrawFn draw, void* arg) {
    ID3D11BlendState* gameBlend = nullptr;
    float gameFactor[4] = {};
    UINT gameMask = 0;
    c->OMGetBlendState(&gameBlend, gameFactor, &gameMask);
    if (!g_sharpStateLogged) {
        g_sharpStateLogged = true;
        D3D11_BLEND_DESC bd{};
        if (gameBlend) gameBlend->GetDesc(&bd);
        g_sharpOwnBlend = bd.RenderTarget[0].BlendEnable != FALSE;
        Log("Windscreen: the game's sharpen draws %s; [dlaa] game_sharpen %s", g_sharpOwnBlend ? "with blending" : "opaque",
            g_sharpOwnBlend ? "only turns it off (0) or leaves it (above 0)" : "scales it");
    }
    bool handled = false;
    if (amount <= 0.001f) {
        handled = CopySharpenSource(c);
        (handled ? g_sharpCopied : g_sharpKept).fetch_add(1, std::memory_order_relaxed);
    } else if (amount < 0.999f && !g_sharpOwnBlend) {
        if (!g_sharpBlend && !g_sharpBlendFailed) {
            ID3D11Device* dev = nullptr;
            c->GetDevice(&dev);
            D3D11_BLEND_DESC bd{};
            D3D11_RENDER_TARGET_BLEND_DESC& rt = bd.RenderTarget[0];
            rt.BlendEnable = TRUE;
            rt.SrcBlend = rt.SrcBlendAlpha = D3D11_BLEND_BLEND_FACTOR;
            rt.DestBlend = rt.DestBlendAlpha = D3D11_BLEND_INV_BLEND_FACTOR;
            rt.BlendOp = rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
            rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            if (!dev || FAILED(dev->CreateBlendState(&bd, &g_sharpBlend))) {
                g_sharpBlendFailed = true;
                Log("Windscreen: the sharpen's blend state could not be made; [dlaa] game_sharpen below 1 is off or on");
            }
            Rel(dev);
        }
        if (g_sharpBlend && CopySharpenSource(c)) {
            const float f[4] = {amount, amount, amount, amount};
            c->OMSetBlendState(g_sharpBlend, f, 0xFFFFFFFF);
            draw(arg);
            c->OMSetBlendState(gameBlend, gameFactor, gameMask);
            handled = true;
            g_sharpBlended.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_sharpKept.fetch_add(1, std::memory_order_relaxed);
        }
    }
    Rel(gameBlend);
    return handled;
}

// The layer shaders that drew after the sharpen, once each: their checksums (x64\qvfr\ps_dump\<checksum>.dxbc).
void NoteLayer(ID3D11PixelShader* ps, int index) {
    static ID3D11PixelShader* seen[32] = {};
    for (auto& s : seen) {
        if (s == ps) return;
        if (s) continue;
        s = ps;
        const uint32_t* h = g_layerSum[index];
        Log("Windscreen: layer pixel shader %p drew after the sharpen: %08x-%08x-%08x-%08x%s", (void*)ps, h[0], h[1], h[2],
            h[3], g_layerGlass[index] ? " (glass)" : "");
        return;
    }
}

int g_curIndex = -1;  // the bound layer shader's slot

void ResetGroup() {
    g_afterSharpen = false;
    g_sharpTarget = nullptr;
    g_curLayer = false;
    g_curGlass = false;
    g_curIndex = -1;
    g_dilDsv = nullptr;
    g_dilFor = nullptr;
    g_dilTried = false;
}

}  // namespace

void OnPixelShaderMade(ID3D11PixelShader* ps, const void* code, size_t len) {
    if (!ps || !code || g_dead.load()) return;
    Protect("CreatePixelShader", [&] {
        // A new shader at the address of a released one.
        for (auto& s : g_sharpPs)
            if (s.load() == ps) s = nullptr;
        for (auto& s : g_layerPs)
            if (s.load() == ps) s = nullptr;
        if (Contains(code, len, "sharp_strength")) {
            bool kept = false;
            for (auto& s : g_sharpPs) {
                ID3D11PixelShader* none = nullptr;
                if (s.compare_exchange_strong(none, ps)) {
                    kept = true;
                    break;
                }
            }
            Log("Windscreen: the game made its sharpen pixel shader (%p, %zu bytes)%s", (void*)ps, len,
                kept ? "" : "; a fifth one, not followed");
        }
        if (Contains(code, len, "VehicleWindscreen")) {
            for (int i = 0; i < kMaxLayerPs; ++i) {
                ID3D11PixelShader* none = nullptr;
                if (g_layerPs[i].compare_exchange_strong(none, ps)) {
                    if (len >= 20) memcpy(g_layerSum[i], (const uint8_t*)code + 4, 16);
                    g_layerGlass[i] = Contains(code, len, "rainControlMap_tex");
                    if (g_layerGlass[i]) g_glassCount.fetch_add(1, std::memory_order_relaxed);
                    g_layerCount.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        }
    });
}

void OnPixelShaderSet(ID3D11PixelShader* ps) {
    if (g_dead.load(std::memory_order_relaxed)) return;
    g_curSharp = IsSharp(ps);
    g_curPs = ps;
    g_curIndex = g_afterSharpen ? LayerIndex(ps) : -1;
    g_curLayer = g_curIndex >= 0;
    g_curGlass = g_curIndex >= 0 && g_layerGlass[g_curIndex];
}

bool OnDraw(ID3D11DeviceContext* c, DrawFn draw, void* arg) {
    if (g_dead.load(std::memory_order_relaxed)) return false;
    if (!g_curSharp && !g_afterSharpen) return false;
    bool handled = false;
    Protect("draw", [&] {
        if (g_curSharp) {
            if (!g_afterSharpen) {
                ID3D11RenderTargetView* rtv = nullptr;
                c->OMGetRenderTargets(1, &rtv, nullptr);
                ID3D11Resource* res = nullptr;
                if (rtv) rtv->GetResource(&res);
                g_sharpTarget = res;
                g_afterSharpen = res != nullptr;
                Rel(res);
                Rel(rtv);
            }
            const float amount = SharpenAmount();
            if (amount < 0.999f) handled = Sharpen(c, amount, draw, arg);
            return;
        }
        if (!g_curLayer) return;
        ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        ID3D11DepthStencilView* dsv = nullptr;
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, &dsv);
        ID3D11Resource* res = nullptr;
        if (rtv[0]) rtv[0]->GetResource(&res);
        const bool layer = dsv && res && res == g_sharpTarget;
        Rel(res);
        if (layer) NoteLayer(g_curPs, g_curIndex);
        if (layer && !g_layersOn.load(std::memory_order_relaxed)) {
            g_layerSkipped.fetch_add(1, std::memory_order_relaxed);
            handled = true;
        } else if (layer) {
            const int radius = DepthRadius();
            if (radius > 0 && !g_curGlass) g_depthOther.fetch_add(1, std::memory_order_relaxed);
            if (radius > 0 && g_curGlass) {
                // The glass's depth: the scene's made farther, once per group (and again for another depth view).
                if (!g_dilTried || g_dilFor != dsv) {
                    D3D11_VIEWPORT vp{};
                    UINT nvp = 1;
                    c->RSGetViewports(&nvp, &vp);
                    g_dilDsv = nvp ? MakeGlassDepth(c, dsv, vp, radius) : nullptr;
                    g_dilFor = dsv;
                    g_dilTried = true;
                }
                if (g_dilDsv) {
                    {
                        clones::RawScope raw;
                        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, g_dilDsv);
                    }
                    draw(arg);
                    {
                        clones::RawScope raw;
                        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, dsv);
                    }
                    handled = true;
                    g_depthDraws.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        for (auto& r : rtv) Rel(r);
        Rel(dsv);
    });
    return handled;
}

void OnFrameStart() {
    if (g_dead.load(std::memory_order_relaxed)) return;
    ResetGroup();
}

void BeginGroup() { ResetGroup(); }

void EndGroup() {
    if (g_dead.load(std::memory_order_relaxed)) return;
    ResetGroup();
}

void SetLayers(bool on) { g_layersOn = on; }
bool GetLayers() { return g_layersOn.load(); }
void SetDepth(int radius) { g_depthSwitch = radius; }
int GetDepth() { return DepthRadius(); }

void LogStats() {
    if (g_dead.load()) return;
    Log("Windscreen: glass's depth radius %d render px: made %llu (not made %llu), glass draws on it %llu, other layer "
        "draws on the game's %llu; layer shaders known %d (glass %d); layers %s (skipped %llu); the game's sharpen at "
        "%.2f: copied (off) %llu, blended %llu, as drawn below 1 %llu",
        DepthRadius(), (unsigned long long)g_depthBuilds.load(), (unsigned long long)g_depthFails.load(),
        (unsigned long long)g_depthDraws.load(), (unsigned long long)g_depthOther.load(), g_layerCount.load(),
        g_glassCount.load(), g_layersOn.load() ? "on" : "OFF (switch)", (unsigned long long)g_layerSkipped.load(),
        SharpenAmount(), (unsigned long long)g_sharpCopied.load(), (unsigned long long)g_sharpBlended.load(),
        (unsigned long long)g_sharpKept.load());
}

void Shutdown() {
    for (DepthScratch& s : g_dep) ReleaseDepth(s);
    Rel(g_dilCs);
    Rel(g_dcb);
    Rel(g_sharpBlend);
    ResetGroup();
}

}  // namespace qvfr::windscreen
