// --dlaa-displays: the displays pass after DLSS SR (shaders/dlaa.hlsl CSDisplays, docs/dlaa-plan.md S2.4) on a real
// D3D11 device, without SteamVR and without the proxy DLL.
// Render size 48 x 40, output 72 x 60 (DLSS SR at 2/3). The game's normal/stencil is a three-slice array; slice 1 (the
// one read, through a one-slice array view as dlaa.cpp makes it) has the displays class 0x88 in a rectangle, the other
// slices have it everywhere (a wrong slice shows up). The current frame is 0.5 with a bright bar (4.0, a digit) inside
// the display; DLSS's result is 2.0 everywhere (an old digit in its history). Checks against a CPU reference, with and
// without a jitter:
// - outside the display the output is DLSS's result;
// - inside, the current frame bilinear at the pixel's position minus the jitter (nothing of DLSS's 2.0 stays);
// - the debug layer stays quiet.
#include <windows.h>
#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dlaa_displays_cs.h"  // kDlaaDisplaysCS

namespace {

constexpr UINT kRW = 48, kRH = 40, kOW = 72, kOH = 60;
constexpr int kDispX0 = 10, kDispX1 = 29, kDispY0 = 8, kDispY1 = 27;  // the display, render pixels (inclusive)
constexpr int kBarX0 = 20, kBarX1 = 21;                                // the bright bar inside it
constexpr float kBg = 0.5f, kBar = 4.0f, kDlss = 2.0f;

// R11G11B10_FLOAT: exact for the values used here (0.5, 2, 4 and their bilinear mixes are checked with a tolerance).
uint32_t F11(float f) {  // positive, normal
    int e;
    const float m = std::frexp(f, &e);  // f = m * 2^e, m in [0.5, 1)
    const int be = e - 1 + 15;
    const uint32_t mant = (uint32_t)std::lround((m * 2.0f - 1.0f) * 64.0f);
    return ((uint32_t)be << 6) | std::min(mant, 63u);
}
uint32_t F10(float f) {
    int e;
    const float m = std::frexp(f, &e);
    const int be = e - 1 + 15;
    const uint32_t mant = (uint32_t)std::lround((m * 2.0f - 1.0f) * 32.0f);
    return ((uint32_t)be << 5) | std::min(mant, 31u);
}
uint32_t Pack(float v) { return F11(v) | (F11(v) << 11) | (F10(v) << 22); }
float Unf11(uint32_t v) {
    const uint32_t e = v >> 6, m = v & 63;
    if (e == 0) return std::ldexp((float)m / 64.0f, -14);
    return std::ldexp(1.0f + (float)m / 64.0f, (int)e - 15);
}

float Cur(int x, int y) {
    x = std::clamp(x, 0, (int)kRW - 1);
    y = std::clamp(y, 0, (int)kRH - 1);
    const bool disp = x >= kDispX0 && x <= kDispX1 && y >= kDispY0 && y <= kDispY1;
    return disp && x >= kBarX0 && x <= kBarX1 ? kBar : kBg;
}
bool Display(int x, int y) { return x >= kDispX0 && x <= kDispX1 && y >= kDispY0 && y <= kDispY1; }

// The shader's rule for output pixel (x, y).
float Expected(UINT x, UINT y, const float jitter[2]) {
    const float rx = ((float)x + 0.5f) * kRW / kOW, ry = ((float)y + 0.5f) * kRH / kOH;
    const int ix = std::min((int)rx, (int)kRW - 1), iy = std::min((int)ry, (int)kRH - 1);
    if (!Display(ix, iy)) return kDlss;
    const float px = rx - jitter[0] - 0.5f, py = ry - jitter[1] - 0.5f;
    const int bx = (int)std::floor(px), by = (int)std::floor(py);
    const float fx = px - bx, fy = py - by;
    const float top = Cur(bx, by) * (1 - fx) + Cur(bx + 1, by) * fx;
    const float bot = Cur(bx, by + 1) * (1 - fx) + Cur(bx + 1, by + 1) * fx;
    return top * (1 - fy) + bot * fy;
}

template <class T>
void Rel(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

}  // namespace

int RunDlaaDisplays() {
    printf("DLAA displays pass smoke (CSDisplays, render %ux%u -> output %ux%u)\n", kRW, kRH, kOW, kOH);
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

    // The game's normal/stencil: R16G16B16A16 typeless, three slices; .w * 65535 = class << 8 | a low byte.
    std::vector<uint16_t> st((size_t)kRW * kRH * 4 * 3);
    for (UINT s = 0; s < 3; ++s)
        for (UINT y = 0; y < kRH; ++y)
            for (UINT x = 0; x < kRW; ++x) {
                const uint32_t cls = s != 1 || Display((int)x, (int)y) ? 0x88 : 0x86;
                uint16_t* p = &st[(((size_t)s * kRH + y) * kRW + x) * 4];
                p[0] = p[1] = p[2] = 0x8000;
                p[3] = (uint16_t)(cls << 8 | 0x6D);
            }
    D3D11_TEXTURE2D_DESC sd{};
    sd.Width = kRW;
    sd.Height = kRH;
    sd.MipLevels = 1;
    sd.ArraySize = 3;
    sd.Format = DXGI_FORMAT_R16G16B16A16_TYPELESS;
    sd.SampleDesc.Count = 1;
    sd.Usage = D3D11_USAGE_DEFAULT;
    sd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    D3D11_SUBRESOURCE_DATA sinit[3];
    for (UINT s = 0; s < 3; ++s) sinit[s] = {&st[(size_t)s * kRW * kRH * 4], kRW * 8, 0};
    ID3D11Texture2D* stencil = nullptr;
    // The current frame (render size) and DLSS's result (output size): R11G11B10.
    std::vector<uint32_t> cur((size_t)kRW * kRH), dl((size_t)kOW * kOH, Pack(kDlss));
    for (UINT y = 0; y < kRH; ++y)
        for (UINT x = 0; x < kRW; ++x) cur[(size_t)y * kRW + x] = Pack(Cur((int)x, (int)y));
    D3D11_TEXTURE2D_DESC cd{};
    cd.Width = kRW;
    cd.Height = kRH;
    cd.MipLevels = 1;
    cd.ArraySize = 1;
    cd.Format = DXGI_FORMAT_R11G11B10_FLOAT;
    cd.SampleDesc.Count = 1;
    cd.Usage = D3D11_USAGE_DEFAULT;
    cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA cinit{cur.data(), kRW * 4, 0};
    D3D11_TEXTURE2D_DESC od = cd;
    od.Width = kOW;
    od.Height = kOH;
    od.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;  // DLSS's scratch out
    D3D11_SUBRESOURCE_DATA oinit{dl.data(), kOW * 4, 0};
    D3D11_TEXTURE2D_DESC rd = od;
    rd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;  // dispTex
    D3D11_TEXTURE2D_DESC stg = od;
    stg.BindFlags = 0;
    stg.Usage = D3D11_USAGE_STAGING;
    stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D *curTex = nullptr, *dlTex = nullptr, *resTex = nullptr, *stgTex = nullptr;
    ID3D11ShaderResourceView *stSrv = nullptr, *curSrv = nullptr, *dlSrv = nullptr;
    ID3D11UnorderedAccessView* resUav = nullptr;
    ID3D11ComputeShader* cs = nullptr;
    ID3D11Buffer* cb = nullptr;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 32;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    sv.Texture2DArray.MipLevels = 1;
    sv.Texture2DArray.FirstArraySlice = 1;
    sv.Texture2DArray.ArraySize = 1;
    bool ok = SUCCEEDED(dev->CreateTexture2D(&sd, sinit, &stencil)) && SUCCEEDED(dev->CreateTexture2D(&cd, &cinit, &curTex)) &&
              SUCCEEDED(dev->CreateTexture2D(&od, &oinit, &dlTex)) && SUCCEEDED(dev->CreateTexture2D(&rd, nullptr, &resTex)) &&
              SUCCEEDED(dev->CreateTexture2D(&stg, nullptr, &stgTex)) &&
              SUCCEEDED(dev->CreateShaderResourceView(stencil, &sv, &stSrv)) &&
              SUCCEEDED(dev->CreateShaderResourceView(curTex, nullptr, &curSrv)) &&
              SUCCEEDED(dev->CreateShaderResourceView(dlTex, nullptr, &dlSrv)) &&
              SUCCEEDED(dev->CreateUnorderedAccessView(resTex, nullptr, &resUav)) &&
              SUCCEEDED(dev->CreateComputeShader(kDlaaDisplaysCS, sizeof(kDlaaDisplaysCS), nullptr, &cs)) &&
              SUCCEEDED(dev->CreateBuffer(&bd, nullptr, &cb));
    if (!ok) printf("  setup failed\n");

    int failures = ok ? 0 : 1;
    struct Case {
        float jitter[2];
        const char* name;
    } cases[] = {{{0.0f, 0.0f}, "no jitter"}, {{0.3f, -0.2f}, "jitter (0.3, -0.2)"}, {{-0.45f, 0.4f}, "jitter (-0.45, 0.4)"}};
    for (const Case& k : cases) {
        if (!ok) break;
        struct {
            uint32_t bits, pad;
            float jitter[2], toRender[2];
            uint32_t renderSize[2];
        } p{0x88, 0, {k.jitter[0], k.jitter[1]}, {(float)kRW / kOW, (float)kRH / kOH}, {kRW, kRH}};
        ctx->UpdateSubresource(cb, 0, nullptr, &p, 0, 0);
        ctx->CSSetShader(cs, nullptr, 0);
        ctx->CSSetConstantBuffers(2, 1, &cb);
        ctx->CSSetShaderResources(2, 1, &stSrv);
        ID3D11ShaderResourceView* in[2] = {dlSrv, curSrv};
        ctx->CSSetShaderResources(3, 2, in);
        ctx->CSSetUnorderedAccessViews(1, 1, &resUav, nullptr);
        ctx->Dispatch((kOW + 7) / 8, (kOH + 7) / 8, 1);
        ID3D11ShaderResourceView* none[3] = {};
        ID3D11UnorderedAccessView* noUav = nullptr;
        ctx->CSSetShaderResources(2, 3, none);
        ctx->CSSetUnorderedAccessViews(1, 1, &noUav, nullptr);
        ctx->CopyResource(stgTex, resTex);
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(ctx->Map(stgTex, 0, D3D11_MAP_READ, 0, &m))) {
            printf("  %s: map failed\n", k.name);
            ++failures;
            continue;
        }
        int bad = 0, inside = 0, cleared = 0, kept = 0;
        float worst = 0;
        for (UINT y = 0; y < kOH; ++y)
            for (UINT x = 0; x < kOW; ++x) {
                const uint32_t v = ((const uint32_t*)((const uint8_t*)m.pData + (size_t)y * m.RowPitch))[x];
                const float got = Unf11(v & 0x7FF), want = Expected(x, y, k.jitter);
                const float err = std::fabs(got - want) / std::max(want, 0.01f);
                worst = std::max(worst, err);
                if (err > 0.04f) {  // R11G11B10 has 6 bits of mantissa: up to ~1.6% per value
                    if (bad < 4) printf("  %s: pixel (%u, %u) %.4f, expected %.4f\n", k.name, x, y, got, want);
                    ++bad;
                }
                const int ix = std::min((int)(((float)x + 0.5f) * kRW / kOW), (int)kRW - 1);
                const int iy = std::min((int)(((float)y + 0.5f) * kRH / kOH), (int)kRH - 1);
                if (Display(ix, iy)) {
                    ++inside;
                    cleared += std::fabs(got - kBg) < 0.01f;
                    kept += std::fabs(got - kDlss) < 0.01f;
                }
            }
        ctx->Unmap(stgTex, 0);
        // Sanity of the scene itself: most of the display away from the bar loses the old digit (0.5).
        const bool sane = cleared * 2 > inside;
        printf("  %s: %d of %u pixels off (worst relative error %.3f); display pixels %d, cleared to the current frame "
               "%d, DLSS kept %d (%s)\n",
               k.name, bad, kOW * kOH, worst, inside, cleared, kept, bad == 0 && sane ? "OK" : "WRONG");
        if (bad || !sane) ++failures;
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
    }
    printf("  debug layer: %s, %llu message(s)\n", debugLayer ? "on" : "not available", (unsigned long long)msgs);
    if (msgs) ++failures;
    Rel(cb);
    Rel(cs);
    Rel(resUav);
    Rel(dlSrv);
    Rel(curSrv);
    Rel(stSrv);
    Rel(stgTex);
    Rel(resTex);
    Rel(dlTex);
    Rel(curTex);
    Rel(stencil);
    Rel(info);
    Rel(ctx);
    Rel(dev);
    printf("DLAA displays pass smoke: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
