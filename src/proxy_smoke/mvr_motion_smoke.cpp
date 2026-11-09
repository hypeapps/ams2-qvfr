// --mvr-motion: the focus motion vectors from outer's (mvr_motion.h, docs/mvr.md B10) on a real D3D11 device, without
// SteamVR and without the proxy DLL.
// A made-up eye: outer's motion vectors (96 x 80) hold a smooth background field and a disk (a near object) with its own
// large motion; the focus view (64 x 64) is a window of it that moves between the last frame and this one. Outer's and
// focus's linear depth tell the disk (1) from the background (50), each at its own pixel centres. The CPU reference works in clip space (the window map
// focus = (sx x + ox w, sy y + oy w) of mvr_pass), the compute pass in uv, so a wrong uv form of the map shows up.
// Checks: away from the disk's edge the result equals the reference (background bilinear, disk exact); in the band
// around the edge each pixel takes its own object's motion (no blending across the silhouette); the debug layer.
#include <windows.h>
#include <d3d11.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mvr_motion.h"

namespace {

uint16_t ToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    const int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t)s;  // tiny values: zero is enough here
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    // round to nearest
    uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
    if (m & 0x1000) ++h;
    return (uint16_t)h;
}

float FromHalf(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : std::ldexp(1.0f + m / 1024.0f, (int)e - 15);
    return s ? -v : v;
}

constexpr UINT kOW = 96, kOH = 80, kFW = 64, kFH = 64;
// Disk in outer uv (centre, radius) and the motions.
constexpr double kCx = 0.47, kCy = 0.52, kR = 0.06;
constexpr double kDiskV[2] = {0.0012, -0.0007};
void BackV(double u, double v, double out[2]) {
    out[0] = 0.05 + 0.02 * (u - 0.5);
    out[1] = -0.03 + 0.015 * (v - 0.5);
}
bool InDisk(double u, double v) { return std::hypot(u - kCx, (v - kCy) * kOH / kOW) < kR; }
double EdgeDist(double u, double v) {  // in outer texels
    return std::fabs(std::hypot(u - kCx, (v - kCy) * kOH / kOW) - kR) * kOW;
}

struct Clip {
    double sx, ox, sy, oy;
};

// B11's window source: this frame's and the last frame's map, the focus image from outer.
qvfr::mvrmotion::Map g_cur, g_prev;
bool FakeWindows(int, qvfr::mvrmotion::Map* cur, qvfr::mvrmotion::Map* prev, bool* fromOuter) {
    *cur = g_cur;
    *prev = g_prev;
    *fromOuter = true;
    return true;
}

std::vector<uint32_t> ReadTexels(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* src) {
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    std::vector<uint32_t> px;
    ID3D11Texture2D* st = nullptr;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &st))) return px;
    ctx->CopyResource(st, src);
    D3D11_MAPPED_SUBRESOURCE m{};
    if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        px.resize((size_t)d.Width * d.Height);
        for (UINT y = 0; y < d.Height; ++y)
            memcpy(&px[(size_t)y * d.Width], (const uint8_t*)m.pData + (size_t)y * m.RowPitch, d.Width * 4);
        ctx->Unmap(st, 0);
    }
    st->Release();
    return px;
}

}  // namespace

int RunMvrMotion() {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    const bool debugLayer = SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                                        D3D11_CREATE_DEVICE_DEBUG, nullptr, 0, D3D11_SDK_VERSION, &dev,
                                                        nullptr, &ctx));
    if (!debugLayer && FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                                D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
        printf("D3D11CreateDevice failed\n");
        return 1;
    }
    ID3D11InfoQueue* info = nullptr;
    if (debugLayer) dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&info);

    const Clip cur{2.4, 0.2, 2.2, -0.1}, prev{2.38, 0.26, 2.2, -0.05};  // the window ~0.42 x 0.45 of outer
    // From focus uv back to outer uv through clip space (the reference's own path).
    auto focusToOuter = [](const Clip& m, double uf, double vf, double& uo, double& vo) {
        const double xf = 2 * uf - 1, yf = 1 - 2 * vf;
        const double xo = (xf - m.ox) / m.sx, yo = (yf - m.oy) / m.sy;
        uo = (xo + 1) / 2;
        vo = (1 - yo) / 2;
    };
    auto outerToFocus = [](const Clip& m, double uo, double vo, double& uf, double& vf) {
        const double xo = 2 * uo - 1, yo = 1 - 2 * vo;
        const double xf = m.sx * xo + m.ox, yf = m.sy * yo + m.oy;
        uf = (xf + 1) / 2;
        vf = (1 - yf) / 2;
    };

    std::vector<uint16_t> mv(kOW * kOH * 2);
    for (UINT y = 0; y < kOH; ++y)
        for (UINT x = 0; x < kOW; ++x) {
            const double u = (x + 0.5) / kOW, v = (y + 0.5) / kOH;
            double m[2];
            if (InDisk(u, v)) {
                m[0] = kDiskV[0];
                m[1] = kDiskV[1];
            } else {
                BackV(u, v, m);
            }
            mv[(y * kOW + x) * 2] = ToHalf((float)m[0]);
            mv[(y * kOW + x) * 2 + 1] = ToHalf((float)m[1]);
        }
    std::vector<float> outerDepth(kOW * kOH);
    for (UINT y = 0; y < kOH; ++y)
        for (UINT x = 0; x < kOW; ++x) outerDepth[y * kOW + x] = InDisk((x + 0.5) / kOW, (y + 0.5) / kOH) ? 1.0f : 50.0f;
    std::vector<float> depth(kFW * kFH);
    for (UINT y = 0; y < kFH; ++y)
        for (UINT x = 0; x < kFW; ++x) {
            double uo, vo;
            focusToOuter(cur, (x + 0.5) / kFW, (y + 0.5) / kFH, uo, vo);
            depth[y * kFW + x] = InDisk(uo, vo) ? 1.0f : 50.0f;
        }

    auto make = [&](UINT w, UINT h, DXGI_FORMAT f, const void* data, UINT pitch) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = w;
        d.Height = h;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = f;
        d.SampleDesc.Count = 1;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA sd{data, pitch, 0};
        ID3D11Texture2D* t = nullptr;
        return SUCCEEDED(dev->CreateTexture2D(&d, &sd, &t)) ? t : nullptr;
    };
    ID3D11Texture2D* outerTex = make(kOW, kOH, DXGI_FORMAT_R16G16_FLOAT, mv.data(), kOW * 4);
    ID3D11Texture2D* depthTex = make(kFW, kFH, DXGI_FORMAT_R32_FLOAT, depth.data(), kFW * 4);
    ID3D11Texture2D* outerDepthTex = make(kOW, kOH, DXGI_FORMAT_R32_FLOAT, outerDepth.data(), kOW * 4);
    if (!outerTex || !depthTex || !outerDepthTex) {
        printf("MVR motion: textures failed\n");
        return 1;
    }

    using qvfr::mvrmotion::FromClip;
    const auto mc = FromClip(cur.sx, cur.ox, cur.sy, cur.oy), mp = FromClip(prev.sx, prev.ox, prev.sy, prev.oy);
    int fails = 0;
    for (int pass = 0; pass < 2; ++pass) {
        const bool nearest = pass == 1;
        ID3D11Texture2D* out = qvfr::mvrmotion::TestRun(ctx, outerTex, outerDepthTex, depthTex, mc, mp, kFW, kFH, nearest);
        if (!out) {
            printf("MVR motion: TestRun failed\n");
            return 1;
        }
        D3D11_TEXTURE2D_DESC d{};
        out->GetDesc(&d);
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* st = nullptr;
        dev->CreateTexture2D(&d, nullptr, &st);
        ctx->CopyResource(st, out);
        D3D11_MAPPED_SUBRESOURCE m{};
        if (!st || FAILED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
            printf("MVR motion: readback failed\n");
            return 1;
        }
        double maxInner = 0, maxBand = 0, maxWindow = 0;
        int inner = 0, band = 0, diskPx = 0;
        for (UINT y = 0; y < kFH; ++y) {
            const uint16_t* row = (const uint16_t*)((const uint8_t*)m.pData + y * m.RowPitch);
            for (UINT x = 0; x < kFW; ++x) {
                const double uf = (x + 0.5) / kFW, vf = (y + 0.5) / kFH;
                double uo, vo;
                focusToOuter(cur, uf, vf, uo, vo);
                const bool disk = InDisk(uo, vo);
                double t[2];
                if (disk) {
                    t[0] = kDiskV[0];
                    t[1] = kDiskV[1];
                    ++diskPx;
                } else {
                    BackV(uo, vo, t);
                }
                double pu, pv;
                outerToFocus(prev, uo - t[0], vo - t[1], pu, pv);
                const double ex = uf - pu, ey = vf - pv;
                const double gx = FromHalf(row[2 * x]), gy = FromHalf(row[2 * x + 1]);
                const double err = std::hypot((gx - ex) * kFW, (gy - ey) * kFH);  // focus px
                // The window's own move alone (motion 0 in outer) for scale.
                double wu, wv;
                outerToFocus(prev, uo, vo, wu, wv);
                maxWindow = std::max(maxWindow, std::hypot((uf - wu) * kFW, (vf - wv) * kFH));
                const double ed = EdgeDist(uo, vo);
                if (ed > 1.5) {
                    maxInner = std::max(maxInner, err);
                    ++inner;
                } else if (ed > 0.75) {
                    maxBand = std::max(maxBand, err);
                    ++band;
                }
            }
        }
        ctx->Unmap(st, 0);
        st->Release();
        // fp16 of the motion (about 3e-5 uv at 0.05) and the background's gradient over one texel when nearest is used.
        const double tolInner = nearest ? 0.08 : 0.02, tolBand = 0.08;
        const bool ok = maxInner <= tolInner && maxBand <= tolBand && diskPx > 50;
        if (!ok) ++fails;
        printf("MVR motion (%s): %d px away from the edge max error %.4f px (limit %.2f), %d px in the edge band max "
               "%.4f px (limit %.2f), disk %d px; the window's own move up to %.2f px: %s\n",
               nearest ? "nearest texel" : "depth-picked bilinear", inner, maxInner, tolInner, band, maxBand, tolBand,
               diskPx, maxWindow, ok ? "ok" : "FAIL");
    }

    // B11: FillFocus writes the same vectors into the game's (typeless) velocity target from outer's copy of the window
    // (StoreOuter), puts back the CS state it changes, and ForFocus then leaves the view as it is.
    {
        g_cur = mc;
        g_prev = mp;
        qvfr::mvrmotion::SetWindowSource(&FakeWindows);
        std::vector<uint32_t> ref;
        if (ID3D11Texture2D* out = qvfr::mvrmotion::TestRun(ctx, outerTex, outerDepthTex, depthTex, mc, mp, kFW, kFH, false))
            ref = ReadTexels(dev, ctx, out);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = kFW;
        td.Height = kFH;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R16G16_TYPELESS;
        td.SampleDesc.Count = 1;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        ID3D11Texture2D* target = nullptr;
        dev->CreateTexture2D(&td, nullptr, &target);
        td.Format = DXGI_FORMAT_R32_FLOAT;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        ID3D11Texture2D* uavTex = nullptr;
        dev->CreateTexture2D(&td, nullptr, &uavTex);
        ID3D11UnorderedAccessView* gameUav = nullptr;
        ID3D11ShaderResourceView* gameSrv = nullptr;
        ID3D11Buffer* gameCb = nullptr;
        D3D11_BUFFER_DESC bd{16, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        if (uavTex) dev->CreateUnorderedAccessView(uavTex, nullptr, &gameUav);
        dev->CreateShaderResourceView(outerDepthTex, nullptr, &gameSrv);
        dev->CreateBuffer(&bd, nullptr, &gameCb);
        bool fillOk = false, refusedOk = false, stateOk = false, forFocusOk = false;
        int same = -1;
        if (target && gameUav && gameSrv && gameCb && !ref.empty()) {
            // The game's CS state while its pass runs.
            ctx->CSSetShaderResources(2, 1, &gameSrv);
            ctx->CSSetConstantBuffers(0, 1, &gameCb);
            ctx->CSSetUnorderedAccessViews(0, 1, &gameUav, nullptr);
            qvfr::mvrmotion::StoreOuter(ctx, 0, outerTex, outerDepthTex, 7, false);
            const char* why = "unset";
            fillOk = qvfr::mvrmotion::FillFocus(ctx, 0, depthTex, target, 7, &why) && !why;
            const char* why2 = nullptr;
            refusedOk = !qvfr::mvrmotion::FillFocus(ctx, 0, depthTex, target, 8, &why2) && why2;  // no copy from frame 8
            ID3D11ShaderResourceView* s[3] = {};
            ID3D11Buffer* b = nullptr;
            ID3D11UnorderedAccessView* u = nullptr;
            ctx->CSGetShaderResources(0, 3, s);
            ctx->CSGetConstantBuffers(0, 1, &b);
            ctx->CSGetUnorderedAccessViews(0, 1, &u);
            stateOk = !s[0] && !s[1] && s[2] == gameSrv && b == gameCb && u == gameUav;
            for (auto* v : s)
                if (v) v->Release();
            if (b) b->Release();
            if (u) u->Release();
            ID3D11ShaderResourceView* none = nullptr;
            ID3D11UnorderedAccessView* noUav = nullptr;
            ctx->CSSetShaderResources(2, 1, &none);
            ctx->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
            const std::vector<uint32_t> got = ReadTexels(dev, ctx, target);
            same = 0;
            for (size_t i = 0; i < got.size() && i < ref.size(); ++i) same += got[i] == ref[i];
            forFocusOk = !qvfr::mvrmotion::ForFocus(ctx, 0, depthTex, nullptr, kFW, kFH, 7, false);
            printf("MVR motion fill (B11): written %s, refused without outer's copy of the frame %s (%s), CS state put "
                   "back %s, target equals the compute pass %d / %u px, ForFocus leaves it %s\n",
                   fillOk ? "ok" : "FAIL", refusedOk ? "ok" : "FAIL", why2 ? why2 : "-", stateOk ? "ok" : "FAIL", same,
                   kFW * kFH, forFocusOk ? "ok" : "FAIL");
        } else {
            printf("MVR motion fill (B11): objects failed\n");
        }
        if (!fillOk || !refusedOk || !stateOk || !forFocusOk || same != (int)(kFW * kFH)) ++fails;
        for (IUnknown* o : {(IUnknown*)target, (IUnknown*)uavTex, (IUnknown*)gameUav, (IUnknown*)gameSrv,
                            (IUnknown*)gameCb})
            if (o) o->Release();
        qvfr::mvrmotion::SetWindowSource(nullptr);
    }
    UINT64 msgs = 0;
    if (info) {
        msgs = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < msgs && i < 8; ++i) {
            SIZE_T len = 0;
            info->GetMessage(i, nullptr, &len);
            std::vector<char> buf(len);
            auto* msg = (D3D11_MESSAGE*)buf.data();
            if (SUCCEEDED(info->GetMessage(i, msg, &len))) printf("  debug layer: %s\n", msg->pDescription);
        }
        info->Release();
    }
    qvfr::mvrmotion::Release();
    outerTex->Release();
    depthTex->Release();
    outerDepthTex->Release();
    ctx->Release();
    dev->Release();
    const bool pass = fails == 0 && msgs == 0;
    printf("MVR motion: %s (debug layer %s, %llu messages)\n", pass ? "PASS" : "FAIL", debugLayer ? "on" : "off",
           (unsigned long long)msgs);
    return pass ? 0 : 1;
}
