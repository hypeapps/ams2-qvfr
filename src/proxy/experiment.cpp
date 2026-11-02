#include "experiment.h"

#include <d3d11.h>

#include <algorithm>
#include <cstdio>
#include <mutex>

#include "common.h"
#include "focus_geometry.h"

namespace qvfr::experiment {

namespace {

// Focus window and frusta: focus_geometry.h (tangent conventions there). All offsets below come from it,
// so the copy position always matches the projection we return.
using Eye = focus::Eye;

struct State {
    bool setupDone = false;
    bool active = false;  // mode != off and setup passed its checks
    ExpMode mode = ExpMode::Off;
    focus::Geometry geo;  // focusW x focusH is the size returned to the game in focus and outer modes
    // pad textures (focus mode)
    ID3D11Device* padDevice = nullptr;  // no reference held, only compared
    ID3D11Texture2D* pad[2] = {};
    DXGI_FORMAT padFormat = DXGI_FORMAT_UNKNOWN;
    bool padFailed = false;
    bool skipLogged = false;
    bool padLogged[2] = {};
    // Submit counters in focus mode with pad=1, so a summary cannot claim "pad on" for frames that went out stretched.
    uint64_t padded = 0, skipped = 0;
    uint64_t runPadded = 0, runSkipped = 0;  // values at MarkRunStart
    const char* skipReason = "";
};

std::mutex g_m;
State g_s;

// The experiment changes what the game renders, so it must keep working when the diagnostics kill-switch
// (g_diagDisabled) trips; it has its own guard and its own fault flag. The proxy is built with /EHa, so a fault
// unwinds the lock_guard inside f and g_m is released.
std::atomic<bool> g_fault{false};
std::atomic<bool> g_engaged{false};  // the game got a replaced size in this VR session

template <class F>
void Safe(const char* where, F&& f) {
    if (g_fault.load(std::memory_order_relaxed)) return;
    unsigned long code = 0;
    __try {
        f();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        g_fault = true;
        Log("ERROR: exception 0x%08lX in the experiment (%s). Experiment stopped for this session; the game keeps what it "
            "already got (size, projection), Submit goes out without pad.", code, where);
    }
}

int Idx(vr::EVREye e) { return e == vr::Eye_Right ? 1 : 0; }

void Setup(vr::IVRSystem* real) {
    if (g_s.setupDone) return;
    g_s.setupDone = true;
    g_s.mode = Cfg().expMode;
    if (g_s.mode == ExpMode::Off || !real) return;
    const float size = Cfg().expFocusSize;
    if (!focus::Compute(real, size, size, Cfg().expCenterX, Cfg().expCenterY, "Experiment", g_s.geo)) {
        Log("Experiment: off (see the line above)");
        return;
    }
    g_s.active = true;
    g_engaged = true;
    const focus::Geometry& g = g_s.geo;
    Log("Experiment: mode %s, SteamVR size (1:1 grid) %u x %u, size returned to the game %u x %u (focus_size %.3f)",
        ExpModeName(g_s.mode), g.gridW, g.gridH, g.focusW, g.focusH, size);
    for (int i = 0; i < 2; ++i) {
        const Eye& e = g.eye[i];
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
    d.Width = g_s.geo.gridW;
    d.Height = g_s.geo.gridH;
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

void Skip(const char* reason) {
    ++g_s.skipped;
    g_s.skipReason = reason;
    if (!g_s.skipLogged) {
        g_s.skipLogged = true;
        Log("Experiment: Submit without pad (%s); the compositor stretches the focus image over the full view", reason);
    }
}

bool FullBounds(const vr::VRTextureBounds_t* b) {
    return !b || (b->uMin == 0.0f && b->vMin == 0.0f && b->uMax == 1.0f && b->vMax == 1.0f);
}

void PadLocked(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds, vr::Texture_t& storage,
               const vr::Texture_t*& out) {
    if (!g_s.active || g_s.mode != ExpMode::Focus || !Cfg().expPad) return;
    if (g_s.padFailed) return Skip("pad texture could not be created");
    if (!tex || tex->eType != vr::API_DirectX || !tex->handle) return Skip("not a D3D11 texture");
    if (!FullBounds(bounds)) return Skip("Submit with texture bounds");
    ID3D11Texture2D* src = nullptr;
    if (FAILED(static_cast<IUnknown*>(tex->handle)->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&src)) || !src)
        return Skip("not an ID3D11Texture2D");
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    if (d.Width != g_s.geo.focusW || d.Height != g_s.geo.focusH || d.SampleDesc.Count != 1 || d.ArraySize != 1) {
        if (!g_s.skipLogged)
            Log("Experiment: game texture %u x %u (samples %u, array %u) is not the returned focus size %u x %u",
                d.Width, d.Height, d.SampleDesc.Count, d.ArraySize, g_s.geo.focusW, g_s.geo.focusH);
        src->Release();
        return Skip("game texture size differs from the returned size");
    }
    int e = Idx(eye);
    ID3D11Device* dev = nullptr;
    src->GetDevice(&dev);
    if (dev && EnsurePad(e, dev, d)) {
        ID3D11DeviceContext* ctx = nullptr;
        dev->GetImmediateContext(&ctx);
        D3D11_BOX box{0, 0, 0, g_s.geo.focusW, g_s.geo.focusH, 1};
        ctx->CopySubresourceRegion(g_s.pad[e], 0, g_s.geo.eye[e].x0, g_s.geo.eye[e].y0, 0, src, 0, &box);
        ctx->Release();
        storage = *tex;
        storage.handle = g_s.pad[e];
        out = &storage;
        ++g_s.padded;
        if (!g_s.padLogged[e]) {
            g_s.padLogged[e] = true;
            Log("Experiment: %s eye goes to the compositor as pad texture %p (%u x %u), game image at pixel (%u, %u)",
                e ? "right" : "left", (void*)g_s.pad[e], g_s.geo.gridW, g_s.geo.gridH, g_s.geo.eye[e].x0,
                g_s.geo.eye[e].y0);
        }
    } else {
        g_s.padFailed = true;
        Skip("pad texture could not be created");
    }
    if (dev) dev->Release();
    src->Release();
}

void ShutdownLocked() {
    std::lock_guard lock(g_m);
    if (g_s.active && g_s.mode == ExpMode::Focus && Cfg().expPad)
        Log("Experiment: session end, %llu Submit padded, %llu without pad%s%s%s", (unsigned long long)g_s.padded,
            (unsigned long long)g_s.skipped, g_s.skipped ? " (last reason: " : "", g_s.skipped ? g_s.skipReason : "",
            g_s.skipped ? ")" : "");
    ReleasePads();
    g_s = State{};
}

}  // namespace

void AdjustSize(vr::IVRSystem* real, uint32_t* w, uint32_t* h) {
    Safe("AdjustSize", [&] {
        std::lock_guard lock(g_m);
        Setup(real);
        if (!g_s.active || !w || !h) return;
        *w = g_s.geo.focusW;
        *h = g_s.geo.focusH;
    });
}

void AdjustProjectionRaw(vr::IVRSystem* real, vr::EVREye eye, float* l, float* r, float* t, float* b) {
    Safe("AdjustProjectionRaw", [&] {
        std::lock_guard lock(g_m);
        Setup(real);
        if (!g_s.active || g_s.mode != ExpMode::Focus || !l || !r || !t || !b) return;
        const Eye& e = g_s.geo.eye[Idx(eye)];
        *l = e.fl;
        *r = e.fr;
        *t = e.ft;
        *b = e.fb;
    });
}

void AdjustProjectionMatrix(vr::IVRSystem* real, vr::EVREye eye, vr::HmdMatrix44_t& m) {
    Safe("AdjustProjectionMatrix", [&] {
        std::lock_guard lock(g_m);
        Setup(real);
        if (!g_s.active || g_s.mode != ExpMode::Focus) return;
        const Eye& e = g_s.geo.eye[Idx(eye)];
        // Only the x/y terms change; the depth rows stay as the runtime built them for the game's near/far and convention.
        m.m[0][0] = 2 / (e.fr - e.fl);
        m.m[0][2] = (e.fr + e.fl) / (e.fr - e.fl);
        m.m[1][1] = 2 / (e.fb - e.ft);
        m.m[1][2] = (e.fb + e.ft) / (e.fb - e.ft);
    });
}

const vr::Texture_t* PrepareSubmit(vr::EVREye eye, const vr::Texture_t* tex, const vr::VRTextureBounds_t* bounds,
                                   vr::Texture_t& storage) {
    const vr::Texture_t* out = tex;
    Safe("PrepareSubmit", [&] {
        std::lock_guard lock(g_m);
        PadLocked(eye, tex, bounds, storage, out);
    });
    return out;
}

bool Engaged() { return g_engaged.load(std::memory_order_relaxed); }

void MarkRunStart() {
    Safe("MarkRunStart", [&] {
        std::lock_guard lock(g_m);
        g_s.runPadded = g_s.padded;
        g_s.runSkipped = g_s.skipped;
    });
}

std::string Describe() {
    char buf[640] = "Experiment: (error)";
    Safe("Describe", [&] {
        std::lock_guard lock(g_m);
        if (!g_s.active) {
            snprintf(buf, sizeof(buf), "Experiment: %s", g_s.mode == ExpMode::Off ? "off" : "inactive (see log)");
            return;
        }
        char pad[200] = "";
        if (g_s.mode == ExpMode::Outer) {
            snprintf(pad, sizeof(pad), " (not used: outer keeps the full projection)");
        } else if (!Cfg().expPad) {
            snprintf(pad, sizeof(pad), ", pad off (pad=0)");
        } else {
            uint64_t p = g_s.padded - g_s.runPadded, s = g_s.skipped - g_s.runSkipped;
            snprintf(pad, sizeof(pad), ", pad: %llu Submit padded, %llu without pad%s%s%s since the run start",
                     (unsigned long long)p, (unsigned long long)s, s ? " (" : "", s ? g_s.skipReason : "", s ? ")" : "");
        }
        const focus::Geometry& g = g_s.geo;
        const Eye& e = g.eye[0];
        snprintf(buf, sizeof(buf),
                 "Experiment: mode %s, SteamVR size %u x %u, returned to the game %u x %u, focus_size %.3f; "
                 "left eye focus l %.4f r %.4f t %.4f b %.4f%s",
                 ExpModeName(g_s.mode), g.gridW, g.gridH, g.focusW, g.focusH, Cfg().expFocusSize, e.fl, e.fr, e.ft, e.fb,
                 pad);
    });
    if (g_fault) return std::string(buf) + " [experiment stopped after an exception, see log]";
    return buf;
}

const char* RunSuffix() {
    const char* s = "";
    Safe("RunSuffix", [&] {
        std::lock_guard lock(g_m);
        if (!g_s.active) return;
        if (g_s.mode == ExpMode::Outer)
            s = "_outer";
        else
            s = Cfg().expPad ? "_focus" : "_focus_nopad";
    });
    return s;
}

void Shutdown() {
    // Not gated by g_fault: the textures must be released and the next VR session starts clean.
    unsigned long code = 0;
    __try {
        ShutdownLocked();
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        Log("ERROR: exception 0x%08lX in experiment::Shutdown", code);
    }
    g_fault = false;
    g_engaged = false;
}

}  // namespace qvfr::experiment
