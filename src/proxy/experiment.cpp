#include "experiment.h"

#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>

#include "common.h"

namespace qvfr::experiment {

namespace {

// Tangent conventions follow the OpenVR projection matrix (DirectX):
//   m00 = 2/(r-l), m02 = (r+l)/(r-l), m11 = 2/(b-t), m12 = (b+t)/(b-t).
// From it, texture column 0 (NDC x = -1) is tangent l and row 0 (NDC y = +1) is tangent b. So in "y up" terms
// b is the upper edge and t the lower edge, whatever the OpenVR names say. All offsets below come from this formula,
// so the copy position always matches the projection we return.
struct Eye {
    float l = 0, r = 0, t = 0, b = 0;      // real frustum (GetProjectionRaw)
    float fl = 0, fr = 0, ft = 0, fb = 0;  // focus frustum, edges on the 1:1 pixel grid
    uint32_t x0 = 0, y0 = 0;               // focus position in the grid (pixels)
};

struct State {
    bool setupDone = false;
    bool active = false;  // mode != off and setup passed its checks
    ExpMode mode = ExpMode::Off;
    uint32_t gridW = 0, gridH = 0;    // SteamVR size = 1:1 grid
    uint32_t focusW = 0, focusH = 0;  // size returned to the game in focus and outer modes
    Eye eye[2];
    // pad textures (focus mode)
    ID3D11Device* padDevice = nullptr;  // no reference held, only compared
    ID3D11Texture2D* pad[2] = {};
    DXGI_FORMAT padFormat = DXGI_FORMAT_UNKNOWN;
    bool padFailed = false;
    bool mismatchLogged = false;
    bool padLogged[2] = {};
};

std::mutex g_m;
State g_s;

int Idx(vr::EVREye e) { return e == vr::Eye_Right ? 1 : 0; }

bool Near(float a, float b) { return std::fabs(a - b) <= 1e-3f; }

void Setup(vr::IVRSystem* real) {
    if (g_s.setupDone) return;
    g_s.setupDone = true;
    g_s.mode = Cfg().expMode;
    if (g_s.mode == ExpMode::Off || !real) return;

    real->GetRecommendedRenderTargetSize(&g_s.gridW, &g_s.gridH);
    if (g_s.gridW < 64 || g_s.gridH < 64) {
        Log("Experiment: SteamVR size %u x %u is not usable, experiment off", g_s.gridW, g_s.gridH);
        return;
    }
    const float size = Cfg().expFocusSize;
    g_s.focusW = std::max(16u, (uint32_t)std::lround(size * g_s.gridW));
    g_s.focusH = std::max(16u, (uint32_t)std::lround(size * g_s.gridH));
    g_s.focusW = std::min(g_s.focusW, g_s.gridW);
    g_s.focusH = std::min(g_s.focusH, g_s.gridH);

    for (int i = 0; i < 2; ++i) {
        vr::EVREye ev = i ? vr::Eye_Right : vr::Eye_Left;
        Eye& e = g_s.eye[i];
        real->GetProjectionRaw(ev, &e.l, &e.r, &e.t, &e.b);
        if (!(e.r > e.l) || !(e.b > e.t)) {
            Log("Experiment: unexpected projection (%s eye) l %.5f r %.5f t %.5f b %.5f, experiment off", i ? "right" : "left",
                e.l, e.r, e.t, e.b);
            return;
        }
        // The game uses GetProjectionMatrix; the replacement only works if it matches the formula above.
        vr::HmdMatrix44_t m = real->GetProjectionMatrix(ev, 0.1f, 100.0f, vr::API_DirectX);
        float m00 = 2 / (e.r - e.l), m02 = (e.r + e.l) / (e.r - e.l), m11 = 2 / (e.b - e.t), m12 = (e.b + e.t) / (e.b - e.t);
        if (!Near(m.m[0][0], m00) || !Near(m.m[0][2], m02) || !Near(m.m[1][1], m11) || !Near(m.m[1][2], m12)) {
            Log("Experiment: GetProjectionMatrix (%s eye) does not match GetProjectionRaw: matrix %.5f %.5f %.5f %.5f, "
                "formula %.5f %.5f %.5f %.5f; experiment off",
                i ? "right" : "left", m.m[0][0], m.m[0][2], m.m[1][1], m.m[1][2], m00, m02, m11, m12);
            return;
        }

        const double densX = g_s.gridW / (double)(e.r - e.l);  // pixels per tangent unit
        const double densY = g_s.gridH / (double)(e.b - e.t);
        const double cx = std::clamp((double)Cfg().expCenterX, (double)e.l, (double)e.r);
        const double cy = std::clamp((double)Cfg().expCenterY, (double)e.t, (double)e.b);
        long x0 = std::lround((cx - e.l) * densX - g_s.focusW / 2.0);
        long y0 = std::lround((e.b - cy) * densY - g_s.focusH / 2.0);
        x0 = std::clamp(x0, 0L, (long)(g_s.gridW - g_s.focusW));
        y0 = std::clamp(y0, 0L, (long)(g_s.gridH - g_s.focusH));
        e.x0 = (uint32_t)x0;
        e.y0 = (uint32_t)y0;
        e.fl = (float)(e.l + x0 / densX);
        e.fr = (float)(e.l + (x0 + g_s.focusW) / densX);
        e.fb = (float)(e.b - y0 / densY);
        e.ft = (float)(e.b - (y0 + g_s.focusH) / densY);
    }
    g_s.active = true;
    Log("Experiment: mode %s, SteamVR size (1:1 grid) %u x %u, size returned to the game %u x %u (focus_size %.3f)",
        ExpModeName(g_s.mode), g_s.gridW, g_s.gridH, g_s.focusW, g_s.focusH, size);
    for (int i = 0; i < 2; ++i) {
        const Eye& e = g_s.eye[i];
        Log("Experiment: %s eye: full l %.5f r %.5f t %.5f b %.5f; focus l %.5f r %.5f t %.5f b %.5f at pixel (%u, %u)%s",
            i ? "right" : "left", e.l, e.r, e.t, e.b, e.fl, e.fr, e.ft, e.fb, e.x0, e.y0,
            g_s.mode == ExpMode::Focus ? "" : " (outer: the game keeps the full projection)");
    }
}

DXGI_FORMAT TypedFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default: return f;
    }
}

void ReleasePads() {
    for (auto*& p : g_s.pad) {
        if (p) p->Release();
        p = nullptr;
    }
    g_s.padDevice = nullptr;
    g_s.padFormat = DXGI_FORMAT_UNKNOWN;
}

// Black full-grid texture with the game texture's format, one per eye. Returns false when it cannot be made.
bool EnsurePad(int e, ID3D11Device* dev, const D3D11_TEXTURE2D_DESC& src) {
    if (g_s.padDevice != dev || g_s.padFormat != src.Format) ReleasePads();
    if (g_s.pad[e]) return true;
    D3D11_TEXTURE2D_DESC d = src;
    d.Width = g_s.gridW;
    d.Height = g_s.gridH;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.SampleDesc = {1, 0};
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    d.CPUAccessFlags = 0;
    d.MiscFlags = 0;
    ID3D11Texture2D* tex = nullptr;
    HRESULT hr = dev->CreateTexture2D(&d, nullptr, &tex);
    if (FAILED(hr) || !tex) {
        Log("Experiment: cannot create the %u x %u pad texture (0x%08lX), focus image goes out as is", d.Width, d.Height,
            (unsigned long)hr);
        return false;
    }
    D3D11_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = TypedFormat(src.Format);
    rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    ID3D11RenderTargetView* rtv = nullptr;
    hr = dev->CreateRenderTargetView(tex, &rv, &rtv);
    if (FAILED(hr) || !rtv) {
        Log("Experiment: cannot clear the pad texture (RTV 0x%08lX, format %d), focus image goes out as is",
            (unsigned long)hr, (int)rv.Format);
        tex->Release();
        return false;
    }
    ID3D11DeviceContext* ctx = nullptr;
    dev->GetImmediateContext(&ctx);
    const float black[4] = {0, 0, 0, 1};
    ctx->ClearRenderTargetView(rtv, black);
    ctx->Release();
    rtv->Release();
    g_s.pad[e] = tex;
    g_s.padDevice = dev;
    g_s.padFormat = src.Format;
    return true;
}

}  // namespace

void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h) {
    std::lock_guard lock(g_m);
    Setup(real);
    if (!g_s.active || !w || !h) return;
    *w = g_s.focusW;
    *h = g_s.focusH;
}

void AdjustProjectionRaw(vr::IVRSystem* real, vr::EVREye eye, float* l, float* r, float* t, float* b) {
    std::lock_guard lock(g_m);
    Setup(real);
    if (!g_s.active || g_s.mode != ExpMode::Focus || !l || !r || !t || !b) return;
    const Eye& e = g_s.eye[Idx(eye)];
    *l = e.fl;
    *r = e.fr;
    *t = e.ft;
    *b = e.fb;
}

void AdjustProjectionMatrix(vr::IVRSystem* real, vr::EVREye eye, vr::HmdMatrix44_t& m) {
    std::lock_guard lock(g_m);
    Setup(real);
    if (!g_s.active || g_s.mode != ExpMode::Focus) return;
    const Eye& e = g_s.eye[Idx(eye)];
    // Only the x/y terms change; the depth rows stay as the runtime built them for the game's near/far and convention.
    m.m[0][0] = 2 / (e.fr - e.fl);
    m.m[0][2] = (e.fr + e.fl) / (e.fr - e.fl);
    m.m[1][1] = 2 / (e.fb - e.ft);
    m.m[1][2] = (e.fb + e.ft) / (e.fb - e.ft);
}

const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, vr::Texture_t& storage) {
    std::lock_guard lock(g_m);
    if (!g_s.active || g_s.mode != ExpMode::Focus || !Cfg().expPad || g_s.padFailed) return tex;
    if (!tex || tex->eType != vr::API_DirectX || !tex->handle) return tex;
    ID3D11Texture2D* src = nullptr;
    if (FAILED(static_cast<IUnknown*>(tex->handle)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&src)) || !src)
        return tex;
    const vr::Texture_t* out = tex;
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    if (d.Width != g_s.focusW || d.Height != g_s.focusH || d.SampleDesc.Count != 1 || d.ArraySize != 1) {
        if (!g_s.mismatchLogged) {
            g_s.mismatchLogged = true;
            Log("Experiment: game texture %u x %u (samples %u, array %u) is not the returned focus size %u x %u; "
                "submitting it as is", d.Width, d.Height, d.SampleDesc.Count, d.ArraySize, g_s.focusW, g_s.focusH);
        }
    } else {
        int e = Idx(eye);
        ID3D11Device* dev = nullptr;
        src->GetDevice(&dev);
        if (dev) {
            if (EnsurePad(e, dev, d)) {
                ID3D11DeviceContext* ctx = nullptr;
                dev->GetImmediateContext(&ctx);
                D3D11_BOX box{0, 0, 0, g_s.focusW, g_s.focusH, 1};
                ctx->CopySubresourceRegion(g_s.pad[e], 0, g_s.eye[e].x0, g_s.eye[e].y0, 0, src, 0, &box);
                ctx->Release();
                storage = *tex;
                storage.handle = g_s.pad[e];
                out = &storage;
                if (!g_s.padLogged[e]) {
                    g_s.padLogged[e] = true;
                    Log("Experiment: %s eye goes to the compositor as pad texture %p (%u x %u), game image at pixel (%u, %u)",
                        e ? "right" : "left", (void*)g_s.pad[e], g_s.gridW, g_s.gridH, g_s.eye[e].x0, g_s.eye[e].y0);
                }
            } else {
                g_s.padFailed = true;
            }
            dev->Release();
        }
    }
    src->Release();
    return out;
}

std::string Describe() {
    std::lock_guard lock(g_m);
    char buf[512];
    if (!g_s.active) {
        snprintf(buf, sizeof(buf), "Experiment: %s", g_s.mode == ExpMode::Off ? "off" : "inactive (see log)");
        return buf;
    }
    const Eye& e = g_s.eye[0];
    snprintf(buf, sizeof(buf),
             "Experiment: mode %s, SteamVR size %u x %u, returned to the game %u x %u, focus_size %.3f; "
             "left eye focus l %.4f r %.4f t %.4f b %.4f%s",
             ExpModeName(g_s.mode), g_s.gridW, g_s.gridH, g_s.focusW, g_s.focusH, Cfg().expFocusSize, e.fl, e.fr, e.ft, e.fb,
             g_s.mode == ExpMode::Focus ? (Cfg().expPad && !g_s.padFailed ? ", pad on" : ", pad off")
                                        : " (not used: outer keeps the full projection)");
    return buf;
}

const char* RunSuffix() {
    std::lock_guard lock(g_m);
    if (!g_s.active) return "";
    return g_s.mode == ExpMode::Focus ? "_focus" : "_outer";
}

void Shutdown() {
    std::lock_guard lock(g_m);
    ReleasePads();
    g_s = State{};
}

}  // namespace qvfr::experiment
