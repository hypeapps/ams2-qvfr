#include "mvr_motion.h"

#include <d3d11.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common.h"
#include "mvr_motion_cs.h"  // kMvrMotionCS (fxc from shaders/mvr_motion.hlsl)

namespace qvfr::mvrmotion {

namespace {

using Microsoft::WRL::ComPtr;

constexpr float kDepthTolerance = 0.02f;  // relative linear depth difference that still blends the 2 x 2 texels

struct Params {
    float cur[4];
    float prev[4];
    float sizes[4];
    float opt[4];
};
static_assert(sizeof(Params) == 64, "cbuffer layout");

struct Tex {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
    UINT w = 0, h = 0;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
};

struct DepthView {
    ID3D11Texture2D* tex = nullptr;  // compared only (the view keeps it alive)
    ComPtr<ID3D11ShaderResourceView> srv;
};

struct Stats {
    uint64_t stored[2] = {}, used[2] = {};
    uint64_t notOuter[2] = {}, noStore[2] = {}, noWindow[2] = {}, failed[2] = {};
};

WindowFn g_src = nullptr;
bool g_on = true;
ID3D11Device* g_dev = nullptr;  // compared only; the resources below belong to it
ComPtr<ID3D11ComputeShader> g_cs;
ComPtr<ID3D11Buffer> g_cb;
bool g_csFailed = false;
Tex g_store[2];  // outer's motion vectors per eye (R16G16_FLOAT, outer's size; only the window part is current)
Tex g_storeDepth[2];  // outer's linear depth per eye (R32_FLOAT, the same part)
uint64_t g_storeFrame[2] = {};
Tex g_out[2];  // the result per focus view
DepthView g_depth[4];
int g_depthNext = 0;
Stats g_st, g_last;
bool g_loggedUse[2] = {}, g_loggedFail = false;

const char* kEye[2] = {"L", "R"};

bool EnsureDevice(ID3D11DeviceContext* c) {
    ComPtr<ID3D11Device> dev;
    c->GetDevice(&dev);
    if (!dev) return false;
    if (dev.Get() != g_dev) {
        Release();
        g_dev = dev.Get();
    }
    if (g_cs) return true;
    if (g_csFailed) return false;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(Params);
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(dev->CreateComputeShader(kMvrMotionCS, sizeof(kMvrMotionCS), nullptr, &g_cs)) ||
        FAILED(dev->CreateBuffer(&bd, nullptr, &g_cb))) {
        Log("MVR motion: the compute shader could not be made; the focus views keep the engine's motion vectors");
        g_cs.Reset();
        g_cb.Reset();
        g_csFailed = true;
        return false;
    }
    return true;
}

bool EnsureTex(Tex& t, UINT w, UINT h, bool uav, DXGI_FORMAT fmt = DXGI_FORMAT_R16G16_FLOAT) {
    if (t.tex && t.w == w && t.h == h && t.fmt == fmt) return true;
    t = Tex{};
    D3D11_TEXTURE2D_DESC d{};
    d.Width = w;
    d.Height = h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0);
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &t.tex)) ||
        FAILED(g_dev->CreateShaderResourceView(t.tex.Get(), nullptr, &t.srv)) ||
        (uav && FAILED(g_dev->CreateUnorderedAccessView(t.tex.Get(), nullptr, &t.uav)))) {
        Log("MVR motion: texture %ux%u could not be made", w, h);
        t = Tex{};
        return false;
    }
    t.w = w;
    t.h = h;
    t.fmt = fmt;
    return true;
}

ID3D11ShaderResourceView* DepthSrv(ID3D11Texture2D* tex) {
    for (DepthView& v : g_depth)
        if (v.tex == tex && v.srv) return v.srv.Get();
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format = DXGI_FORMAT_R32_FLOAT;
    sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    DepthView& v = g_depth[g_depthNext];
    g_depthNext = (g_depthNext + 1) % (int)std::size(g_depth);
    v = DepthView{};
    if (FAILED(g_dev->CreateShaderResourceView(tex, &sd, &v.srv))) return nullptr;
    v.tex = tex;
    return v.srv.Get();
}

UINT TexW(ID3D11Texture2D* t, UINT* h) {
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    *h = d.Height;
    return d.Width;
}

void Run(ID3D11DeviceContext* c, ID3D11ShaderResourceView* outer, ID3D11ShaderResourceView* outerDepth, UINT ow,
         UINT oh, ID3D11ShaderResourceView* depth, ID3D11UnorderedAccessView* out, UINT w, UINT h, const Map& cur, const Map& prev, bool nearest) {
    const Params p = {{cur.su, cur.ou, cur.sv, cur.ov},
                      {prev.su, prev.ou, prev.sv, prev.ov},
                      {(float)w, (float)h, (float)ow, (float)oh},
                      {kDepthTolerance, nearest ? 1.0f : 0.0f, 0, 0}};
    c->UpdateSubresource(g_cb.Get(), 0, nullptr, &p, 0, 0);
    ID3D11ShaderResourceView* srv[3] = {outer, depth, outerDepth};
    c->CSSetShader(g_cs.Get(), nullptr, 0);
    ID3D11Buffer* cb = g_cb.Get();
    c->CSSetConstantBuffers(0, 1, &cb);
    c->CSSetShaderResources(0, 3, srv);
    c->CSSetUnorderedAccessViews(0, 1, &out, nullptr);
    c->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    ID3D11ShaderResourceView* noSrv[3] = {};
    ID3D11UnorderedAccessView* noUav = nullptr;
    c->CSSetShaderResources(0, 3, noSrv);
    c->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    c->CSSetShader(nullptr, nullptr, 0);
}

float Half(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(1.0f + m / 1024.0f, (int)e - 15);
    return s ? -v : v;
}

ComPtr<ID3D11Texture2D> Staging(ID3D11DeviceContext* c, ID3D11Texture2D* src) {
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    ComPtr<ID3D11Texture2D> s;
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &s))) return nullptr;
    c->CopyResource(s.Get(), src);
    return s;
}

// Shot frame: this module's vectors (outer's geometry) against the engine's for the focus view (its velocity pass draws
// focus's own geometry): the rounding of focus's matrices plus the resampling at silhouettes, in focus pixels.
void Compare(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* ours, ID3D11Texture2D* game, UINT w, UINT h,
             bool fromOuter) {
    ComPtr<ID3D11Texture2D> a = Staging(c, ours), b = Staging(c, game);
    if (!a || !b) return;
    D3D11_MAPPED_SUBRESOURCE ma{}, mb{};
    if (FAILED(c->Map(a.Get(), 0, D3D11_MAP_READ, 0, &ma))) return;
    if (FAILED(c->Map(b.Get(), 0, D3D11_MAP_READ, 0, &mb))) {
        c->Unmap(a.Get(), 0);
        return;
    }
    std::vector<float> d;
    d.reserve((size_t)w * h);
    double sumGame = 0, sumOurs = 0;
    uint64_t over025 = 0, over1 = 0, nan = 0;
    for (UINT y = 0; y < h; ++y) {
        const uint16_t* ra = (const uint16_t*)((const uint8_t*)ma.pData + (size_t)y * ma.RowPitch);
        const uint16_t* rb = (const uint16_t*)((const uint8_t*)mb.pData + (size_t)y * mb.RowPitch);
        for (UINT x = 0; x < w; ++x) {
            const float ax = Half(ra[2 * x]) * w, ay = Half(ra[2 * x + 1]) * h;
            const float bx = Half(rb[2 * x]) * w, by = Half(rb[2 * x + 1]) * h;
            if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(bx) || !std::isfinite(by)) {
                ++nan;
                continue;
            }
            const float e = std::hypot(ax - bx, ay - by);
            d.push_back(e);
            sumGame += std::hypot(bx, by);
            sumOurs += std::hypot(ax, ay);
            if (e > 0.25f) ++over025;
            if (e > 1.0f) ++over1;
        }
    }
    c->Unmap(a.Get(), 0);
    c->Unmap(b.Get(), 0);
    if (d.empty()) return;
    const size_t n = d.size();
    auto pct = [&](double q) {
        const size_t k = std::min(n - 1, (size_t)(q * (double)(n - 1)));
        std::nth_element(d.begin(), d.begin() + (ptrdiff_t)k, d.end());
        return d[k];
    };
    const float p50 = pct(0.5), p90 = pct(0.9), p99 = pct(0.99), p999 = pct(0.999);
    const float mx = *std::max_element(d.begin(), d.end());
    Log("MVR motion shot (focus %s, %ux%u, image %s): ours vs the engine's motion vectors, difference in focus px: "
        "median %.3f, "
        "p90 %.3f, p99 %.3f, p99.9 %.3f, max %.1f; > 0.25 px %.2f%%, > 1 px %.2f%%; mean length ours %.3f, engine's "
        "%.3f px; non-finite %llu",
        kEye[eye], w, h, fromOuter ? "from outer" : "focus's own", p50, p90, p99, p999, mx, 100.0 * over025 / n, 100.0 * over1 / n, sumOurs / n, sumGame / n,
        (unsigned long long)nan);
}

}  // namespace

Map FromClip(double sx, double ox, double sy, double oy) {
    Map m;
    m.su = (float)sx;
    m.ou = (float)((1 - sx + ox) / 2);
    m.sv = (float)sy;
    m.ov = (float)((1 - sy - oy) / 2);
    return m;
}

void SetWindowSource(WindowFn fn) { g_src = fn; }
void SetEnabled(bool on) { g_on = on; }
bool Enabled() { return g_on; }

void StoreOuter(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* motion, ID3D11Texture2D* depth, uint64_t frame,
                bool shot) {
    if (eye < 0 || eye > 1 || !motion || !depth || !g_src || (!g_on && !shot)) return;
    Map cur, prev;
    bool fromOuter = false;
    if (!g_src(eye, &cur, &prev, &fromOuter) || cur.su <= 0 || cur.sv <= 0) return;
    if (!EnsureDevice(c)) return;
    UINT h = 0;
    const UINT w = TexW(motion, &h);
    UINT dh = 0;
    if (TexW(depth, &dh) != w || dh != h) return;
    if (!EnsureTex(g_store[eye], w, h, false) || !EnsureTex(g_storeDepth[eye], w, h, false, DXGI_FORMAT_R32_FLOAT))
        return;
    // The window in outer's texels (focus uv 0..1), with a margin for the 2 x 2 footprint.
    auto range = [](float s, float o, UINT n, UINT& lo, UINT& hi) {
        const double a = (0 - o) / s * n - 2, b = (1 - o) / s * n + 2;
        lo = (UINT)std::clamp(std::floor(a), 0.0, (double)n);
        hi = (UINT)std::clamp(std::ceil(b), 0.0, (double)n);
    };
    D3D11_BOX box{0, 0, 0, 0, 0, 1};
    range(cur.su, cur.ou, w, box.left, box.right);
    range(cur.sv, cur.ov, h, box.top, box.bottom);
    if (box.right <= box.left || box.bottom <= box.top) return;
    c->CopySubresourceRegion(g_store[eye].tex.Get(), 0, box.left, box.top, 0, motion, 0, &box);
    c->CopySubresourceRegion(g_storeDepth[eye].tex.Get(), 0, box.left, box.top, 0, depth, 0, &box);
    g_storeFrame[eye] = frame;
    ++g_st.stored[eye];
}

ID3D11Texture2D* ForFocus(ID3D11DeviceContext* c, int eye, ID3D11Texture2D* depth, ID3D11Texture2D* motion, UINT w,
                          UINT h, uint64_t frame, bool shot) {
    if (eye < 0 || eye > 1 || !g_src || (!g_on && !shot)) return nullptr;
    Map cur, prev;
    bool fromOuter = false;
    if (!g_src(eye, &cur, &prev, &fromOuter) || cur.su <= 0 || cur.sv <= 0) {
        ++g_st.noWindow[eye];
        return nullptr;
    }
    const bool use = g_on && fromOuter;
    if (!use && !shot) {
        ++g_st.notOuter[eye];
        return nullptr;
    }
    if (g_storeFrame[eye] != frame || !g_store[eye].tex) {
        ++g_st.noStore[eye];
        return nullptr;
    }
    ID3D11Texture2D* result = nullptr;
    [&] {
        if (!depth || !EnsureDevice(c) || !EnsureTex(g_out[eye], w, h, true)) return;
        ID3D11ShaderResourceView* ds = DepthSrv(depth);
        if (!ds) return;
        Run(c, g_store[eye].srv.Get(), g_storeDepth[eye].srv.Get(), g_store[eye].w, g_store[eye].h, ds,
            g_out[eye].uav.Get(), w, h, cur, prev, false);
        result = g_out[eye].tex.Get();
    }();
    if (!result) {
        ++g_st.failed[eye];
        if (!g_loggedFail) {
            g_loggedFail = true;
            Log("MVR motion: focus %s: no result (device, texture or depth view); the engine's motion vectors",
                kEye[eye]);
        }
        return nullptr;
    }
    if (shot && motion) Compare(c, eye, result, motion, w, h, fromOuter);
    if (!use) {
        if (!fromOuter) ++g_st.notOuter[eye];
        return nullptr;
    }
    ++g_st.used[eye];
    if (!g_loggedUse[eye]) {
        g_loggedUse[eye] = true;
        Log("MVR motion: focus %s gets outer's motion vectors (%ux%u from outer %ux%u; window u = %.5f u_o + %.5f, "
            "v = %.5f v_o + %.5f)",
            kEye[eye], w, h, g_store[eye].w, g_store[eye].h, cur.su, cur.ou, cur.sv, cur.ov);
    }
    return result;
}

std::string Describe() {
    char buf[96];
    snprintf(buf, sizeof(buf), "MVR motion %s (used L %llu / R %llu)", g_on ? "on" : "off",
             (unsigned long long)g_st.used[0], (unsigned long long)g_st.used[1]);
    return buf;
}

void LogStats(bool totals) {
    const Stats& a = g_st;
    Stats d = a;
    if (!totals)
        for (int e = 0; e < 2; ++e) {
            d.stored[e] -= g_last.stored[e];
            d.used[e] -= g_last.used[e];
            d.notOuter[e] -= g_last.notOuter[e];
            d.noStore[e] -= g_last.noStore[e];
            d.noWindow[e] -= g_last.noWindow[e];
            d.failed[e] -= g_last.failed[e];
        }
    g_last = a;
    uint64_t any = 0;
    for (int e = 0; e < 2; ++e) any += d.stored[e] + d.used[e] + d.notOuter[e] + d.noStore[e] + d.noWindow[e] + d.failed[e];
    if (!any && !totals) return;
    Log("MVR motion%s (%s): focus motion vectors from outer L %llu / R %llu (outer stored L %llu / R %llu); the "
        "engine's: forward not from outer L %llu / R %llu, no outer copy in this frame L %llu / R %llu, no window "
        "L %llu / R %llu, failed L %llu / R %llu",
        totals ? " (session total)" : "", g_on ? "on" : "off", (unsigned long long)d.used[0],
        (unsigned long long)d.used[1], (unsigned long long)d.stored[0], (unsigned long long)d.stored[1],
        (unsigned long long)d.notOuter[0], (unsigned long long)d.notOuter[1], (unsigned long long)d.noStore[0],
        (unsigned long long)d.noStore[1], (unsigned long long)d.noWindow[0], (unsigned long long)d.noWindow[1],
        (unsigned long long)d.failed[0], (unsigned long long)d.failed[1]);
}

void Release() {
    g_cs.Reset();
    g_cb.Reset();
    g_csFailed = false;
    for (Tex& t : g_store) t = Tex{};
    for (Tex& t : g_storeDepth) t = Tex{};
    for (Tex& t : g_out) t = Tex{};
    for (DepthView& v : g_depth) v = DepthView{};
    g_storeFrame[0] = g_storeFrame[1] = 0;
    g_dev = nullptr;
}

ID3D11Texture2D* TestRun(ID3D11DeviceContext* c, ID3D11Texture2D* outerMotion, ID3D11Texture2D* outerDepth,
                         ID3D11Texture2D* focusDepth, const Map& cur, const Map& prev, UINT w, UINT h, bool nearestOnly) {
    if (!EnsureDevice(c) || !EnsureTex(g_out[0], w, h, true)) return nullptr;
    ComPtr<ID3D11ShaderResourceView> os;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format = DXGI_FORMAT_R16G16_FLOAT;
    sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    if (FAILED(g_dev->CreateShaderResourceView(outerMotion, &sd, &os))) return nullptr;
    ID3D11ShaderResourceView* ds = DepthSrv(focusDepth);
    ComPtr<ID3D11ShaderResourceView> ods;
    sd.Format = DXGI_FORMAT_R32_FLOAT;
    if (!ds || FAILED(g_dev->CreateShaderResourceView(outerDepth, &sd, &ods))) return nullptr;
    UINT oh = 0;
    const UINT ow = TexW(outerMotion, &oh);
    Run(c, os.Get(), ods.Get(), ow, oh, ds, g_out[0].uav.Get(), w, h, cur, prev, nearestOnly);
    return g_out[0].tex.Get();
}

}  // namespace qvfr::mvrmotion
